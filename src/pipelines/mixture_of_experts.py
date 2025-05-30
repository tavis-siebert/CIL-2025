import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cache import load_embeddings
from utils import apply_inverse_label_mapping, apply_label_mapping

from .base import BasePipeline

logger = logging.getLogger(__name__)


class MoE(nn.Module):
    # (Copy or import the existing SentimentMoE implementation here)
    def __init__(self, expert_configs: List[DictConfig], out_size=3):
        super().__init__()

        self.processors = nn.ModuleDict()

        total_fusion_input_dim = 0
        for expert in expert_configs:
            name = expert.embed_model
            self.processors[name] = nn.Sequential(
                nn.LazyLinear(expert.processor_output_dim),
                nn.ReLU(),
            )
            total_fusion_input_dim += expert.processor_output_dim

        self.gate = nn.Sequential(
            nn.Linear(total_fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(expert_configs)),
        )
        # uniform init
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.out_features == len(expert_configs):
                    nn.init.constant_(module.weight, 1e-4)
                    nn.init.constant_(module.bias, 0.0)

        self.fusion = nn.Sequential(
            nn.Linear(total_fusion_input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, out_size),
        )

    def forward(self, expert_inputs, temp=1.0):  # temp left if someone wants to play w/ it in the future
        # NOTE: refactor the commented lines if you want to use the original architecture,
        # but it feels weird to use raw-embeddings if they're not in the same feature space (i.e. pre-processor())
        # ---------------------
        # gate_inputs, processor_outputs = [], []
        processor_outputs = []
        for name, expert_embed in expert_inputs.items():
            # gate_inputs.append(expert_embed)
            processor_outputs.append(self.processors[name](expert_embed))

        # gate_input = torch.cat(gate_inputs, dim=1)
        # expert_weights = self.gate(gate_input)
        expert_weights = torch.softmax(self.gate(torch.cat(processor_outputs, dim=1)) / temp, dim=1)

        weighted_expert_outputs = []
        for i, name in enumerate(expert_inputs):
            weighted_expert_outputs.append(processor_outputs[i] * expert_weights[:, i].unsqueeze(1))

        fusion_input = torch.cat(weighted_expert_outputs, dim=1)
        logits = self.fusion(fusion_input)

        return logits, expert_weights


class MoEModel(BasePipeline):
    """
    Pipeline wrapping a Dynamic Mixture-of-Experts model for sentiment analysis.
    """

    def __init__(
        self,
        config: DictConfig | ListConfig,
        device: str | torch.device | None = None,
        output_dir: str = "output",
        debug: bool = False,
    ):
        super().__init__(config, device, output_dir, debug)

        # Expert names and embeds
        self.expert_names = []
        self.expert_embeddings = {}
        for expert in self.config.experts:
            # save names of experts for accessing embeds / layers later
            name = expert.embed_model
            self.expert_names.append(name)
            # save embeds for each expert
            embed_pipeline = expert.embed_pipeline
            embed_file = f"embeddings_{expert.embed_type}.npz" if embed_pipeline == "huggingface" else "embeddings.npz"
            self.expert_embeddings[name] = load_embeddings(embed_pipeline, name, embed_file)

        # Label mappings
        if config.mode == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
            out_size = 1
        elif config.mode == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
            out_size = 3
        else:
            raise ValueError(f"Unknown label mapping: {config.mode}")

        # MoE model
        self.MoE = MoE(self.config.experts, out_size).to(self.device)

        # 5. Training hyperparameters from config
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.patience = config.patience
        self.entropy_coeff = config.entropy_coeff

    def train(
        self,
        train_sentences: pd.Series,
        train_labels: pd.Series,
        val_sentences: pd.Series,
        val_labels: pd.Series,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Prepare embeddings for each expert
        train_embeddings, val_embeddings = {}, {}
        for name, embeds in self.expert_embeddings.items():
            train_embeddings[name] = (
                torch.from_numpy(embeds["train_embeddings"][train_sentences.index]).float().to(self.device)
            )
            val_embeddings[name] = (
                torch.from_numpy(embeds["train_embeddings"][val_sentences.index]).float().to(self.device)
            )

        # Encode labels
        train_labels = apply_label_mapping(train_labels, self.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.label_mapping)
        if self.config.mode == "classification":
            train_labels = torch.from_numpy(train_labels.values).long().to(self.device)
            val_labels = torch.from_numpy(val_labels.values).long().to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            train_labels = torch.from_numpy(train_labels.values).float().unsqueeze(1).to(self.device)
            val_labels = torch.from_numpy(val_labels.values).float().unsqueeze(1).to(self.device)
            criterion = nn.L1Loss()

        # Build DataLoaders
        train_dataset = TensorDataset(*list(train_embeddings.values()), train_labels)
        val_dataset = TensorDataset(*list(val_embeddings.values()), val_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer, scheduler
        optimizer = AdamW(self.MoE.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_score = 0.0
        patience_counter = 0

        # --- Training starts ----
        logger.info("Starting Training")
        for epoch in range(self.num_epochs):
            # Train epoch
            self.MoE.train()

            samples_count_train = 0
            train_loss = 0.0
            expert_weights_sum_train = torch.zeros(len(self.expert_names)).to(self.device)

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                *expert_tensors, labels = batch
                expert_inputs = {name: tensor for name, tensor in zip(self.expert_names, expert_tensors)}

                optimizer.zero_grad()
                logits, expert_weights = self.MoE(expert_inputs)

                """
                #TODO add diversity loss?
                weights_matrix = torch.stack([ew for ew in expert_weights], dim=0)
                covariance = torch.cov(weights_matrix)
                diversity_loss = torch.norm(covariance, p="fro")
                loss += 0.01 * diversity_loss
                """
                loss = criterion(logits, labels)
                entropy_reg = self.entropy_coeff * (expert_weights * torch.log(expert_weights)).sum(dim=1).mean()
                loss += entropy_reg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.MoE.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                expert_weights_sum_train += expert_weights.sum(dim=0).detach()
                samples_count_train += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_expert_weights_train = expert_weights_sum_train / samples_count_train

            # Validation epoch
            self.MoE.eval()
            samples_count_val = 0
            expert_weights_sum_val = torch.zeros(len(self.expert_names)).to(self.device)
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    *expert_tensors, labels = batch
                    expert_inputs = {name: tensor for name, tensor in zip(self.expert_names, expert_tensors)}
                    preds, expert_weights = self.MoE(expert_inputs)
                    if self.config.mode == "classification":
                        preds = preds.argmax(dim=1)
                    else:
                        preds = preds.round().clip(-1, 1)
                    samples_count_val += labels.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
                    expert_weights_sum_val += expert_weights.sum(dim=0).detach()

            all_preds, all_labels = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
            mae = torch.abs(all_preds.float() - all_labels.float()).mean().item()
            val_score = 0.5 * (2 - mae)

            scheduler.step(val_score)
            avg_expert_weights_val = expert_weights_sum_val / samples_count_val

            # Log metrics
            logger.info(f"Avg Train Loss {avg_train_loss}")
            logger.info(f"Avg Val Score: {val_score}")

            train_weights_str = "\n".join(
                [f"{name} {avg_expert_weights_train[i]:.3f}" for i, name in enumerate(self.expert_names)]
            )
            logger.info(f"Train Expert Weights:\n{train_weights_str}")
            val_weights_str = "\n".join(
                [f"{name} {avg_expert_weights_val[i]:.3f}" for i, name in enumerate(self.expert_names)]
            )
            logger.info(f"Val Expert Weights:\n{val_weights_str}")

            # Early stopping & checkpointing
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
                torch.save(self.MoE.state_dict(), os.path.join(self.output_dir, "models/best_moe.pt"))
                logger.info("Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping after epoch {epoch + 1}")
                    break

        # --- Training ended ----
        logger.info("Training ended.\nStarting Testing")

        # Load best model
        logger.info("Loading best model for inference")
        try:
            self.MoE.load_state_dict(torch.load(os.path.join(self.output_dir, "models/best_moe.pt")))
        except FileNotFoundError:
            logger.info(
                "No best model checkpoint found. Using current model state (likely from last epoch of training if training just finished)."
            )

        # Final predictions on train/val for reporting
        train_preds = self.preds_to_series(self.predict_tensor(train_embeddings, 256), train_sentences.index)
        val_preds = self.preds_to_series(self.predict_tensor(val_embeddings, 256), val_sentences.index)
        return train_preds, val_preds

    @torch.no_grad()
    def predict_tensor(
        self, embeds_dict: dict[str, torch.Tensor], inference_batch_size: int | None = None
    ) -> np.ndarray:
        def compute_preds(embeds_dict_slice):
            out, _ = self.MoE(embeds_dict_slice)
            if self.config.mode == "classification":
                return out.argmax(dim=1).detach().cpu().numpy()
            return out.squeeze().detach().cpu().numpy()

        self.MoE.eval()
        num_samples = len(list(embeds_dict.values())[0])
        if inference_batch_size is None or inference_batch_size >= num_samples:
            logger.warning("Large batch sizes might result in out-of-memory errors if there are many experts")
            return compute_preds(embeds_dict)

        preds = []
        for i in range(0, num_samples, inference_batch_size):
            end = i + inference_batch_size
            batch_expert_embeds_dict = {
                name: embed[i:end]  # slice cached embeddings
                for name, embed in embeds_dict.items()
            }
            batch_preds = compute_preds(batch_expert_embeds_dict)
            preds.extend(batch_preds.tolist())
        return np.array(preds)

    def preds_to_series(self, preds, index):
        preds = pd.Series(preds, index=index)
        if self.config.mode == "regression":
            preds = preds.round().clip(-1, 1).astype(int)
        preds = apply_inverse_label_mapping(preds, self.label_mapping)
        return preds

    def predict(self, sentences: pd.Series) -> pd.Series:
        test_embeddings = {
            name: torch.from_numpy(embeds["test_embeddings"][sentences.index]).float().to(self.device)
            for name, embeds in self.expert_embeddings.items()
        }
        preds = self.preds_to_series(self.predict_tensor(test_embeddings, 256), sentences.index)
        return preds
