"""Module for MLP head model pipeline."""

import logging

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cache import load_embeddings
from utils import apply_inverse_label_mapping, apply_label_mapping

from .base import BasePipeline

logger = logging.getLogger(__name__)


class MLPHeadModel(BasePipeline):
    """Implements a linear head model on top of embeddings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        embeds_file = "embeddings.npz"
        if self.config.embed_pipeline == "huggingface":
            embeds_file = f"embeddings_{self.config.embed_type}.npz"
        self.embeddings = load_embeddings(self.config.embed_pipeline, self.config.embed_model, embeds_file)

        # model
        if self.config.mode == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
            out_size = 1
        elif self.config.mode == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
            out_size = 3
        else:
            raise ValueError(f"Unknown label mapping: {self.config.mode}")

        self.classifier = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_size),
        ).to(self.device)

    def train(self, train_sentences, train_labels, val_sentences, val_labels, **kwargs):
        embeddings = self.embeddings["train_embeddings"]

        train_embeddings = torch.from_numpy(embeddings[train_sentences.index]).float().to(self.device)
        val_embeddings = torch.from_numpy(embeddings[val_sentences.index]).float().to(self.device)

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

        train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4, weight_decay=0.01)

        best_score = 0.0
        patience_counter = 0
        for epoch in range(self.config.num_epochs):
            # train epoch
            self.classifier.train()
            train_loss = 0.0
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                optimizer.zero_grad()
                pred = self.classifier(x_batch)
                loss = criterion(pred, y_batch)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(train_loader)

            # eval
            self.classifier.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    pred = self.classifier(x_batch)
                    if self.config.mode == "classification":
                        pred = pred.argmax(dim=1)
                    else:
                        pred = pred.round().clip(-1, 1)
                    all_preds.append(pred.cpu())
                    all_labels.append(y_batch.cpu())

            all_preds, all_labels = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
            mae = torch.abs(all_preds.float() - all_labels.float()).mean().item()
            val_score = 0.5 * (2 - mae)

            # log metrics
            logger.info(f"Avg Train Loss {avg_train_loss}")
            logger.info(f"Avg Val Score: {val_score}")

            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                # torch.save(self.classifier.state_dict(), os.path.join(self.output_dir, f"models/best_mlp_head_{torch.save(self.classifier.state_dict(), os.path.join(self.output_dir, f"models/best_mlp_head_{self.config.embed_model}.pt"))}.pt"))
                logger.info("Best score")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping after epoch {epoch + 1}")
                    break

        logger.info(f"Best Validation Score: {best_score}")

        logger.info("Training ended.\nStarting Testing")
        train_predictions = self.preds_to_series(self.predict_tensor(train_embeddings), train_sentences.index)
        val_predictions = self.preds_to_series(self.predict_tensor(val_embeddings), val_sentences.index)

        return train_predictions, val_predictions

    def preds_to_series(self, preds, index):
        preds = pd.Series(preds, index=index)
        if self.config.mode == "regression":
            preds = preds.round().clip(-1, 1).astype(int)
        preds = apply_inverse_label_mapping(preds, self.label_mapping)
        return preds

    @torch.no_grad()
    def predict_tensor(self, embeds):
        self.classifier.eval()
        preds = self.classifier(embeds)
        if self.config.mode == "classification":
            return preds.argmax(dim=1).detach().cpu().numpy()
        return preds.squeeze().detach().cpu().numpy()

    def predict(self, sentences):
        embeddings = self.embeddings["test_embeddings"]
        test_embeddings = torch.from_numpy(embeddings[sentences.index]).float().to(self.device)
        preds = self.preds_to_series(self.predict_tensor(test_embeddings), sentences.index)
        return preds
