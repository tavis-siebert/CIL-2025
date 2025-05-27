import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import confusion_matrix
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import (
    apply_inverse_label_mapping,
    apply_label_mapping,
    split_indices,
)

from .base import BasePipeline

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


def is_wandb_logging():
    return wandb is not None and wandb.run is not None


class WeightedLossTrainerConfig(TrainingArguments):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights


# reference: https://huggingface.co/docs/transformers/trainer
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set class weights
        self._class_weights = None
        if self.args.class_weights is not None:
            if self.args.class_weights == "auto":
                label_counts = pd.Series(self.train_dataset["label"]).value_counts().sort_index()
                class_weights = len(self.train_dataset) / (len(label_counts) * label_counts)
                self._class_weights = list(class_weights)
            else:
                self._class_weights = self.args.class_weights
            logger.info(f"Using class weights: {self._class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        if self._class_weights is not None:
            class_weights = torch.tensor(self._class_weights, device=model.device)
        else:
            class_weights = None
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class FinetunedClassifier(BasePipeline):
    """Finetuned sentiment classifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        label2id = OmegaConf.to_container(self.config.label_mapping)
        id2label = {v: k for k, v in label2id.items()}

        # load the model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.pretrained_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            **self.config.model,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # allows to load models with different head size
        )
        model = model.to(self.device)
        logger.info(f"Loaded model: {self.config.model.pretrained_model_name_or_path}\n{model}")
        logger.info(f"Using device: {model.device}")

        # reduce parameters to finetune
        if "freeze" in self.config:
            # freeze specified modules
            for module_path in self.config.freeze:
                for param in model.get_submodule(module_path).parameters():
                    param.requires_grad = False
            logger.info(f"Froze parameters of modules: {self.config.freeze}")
        elif "peft" in self.config:
            # add PEFT adapter to model
            peft_config = OmegaConf.to_container(self.config.peft)
            model = get_peft_model(model, LoraConfig(**peft_config))
            logger.info(f"Using PEFT adapter: {peft_config}")

        # print model summary
        if isinstance(model, PeftModel):
            n_params_trainable, n_params_total = model.get_nb_trainable_parameters()
        else:
            n_params_total = sum(p.numel() for p in model.parameters())
            n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Number of trainable parameters: {n_params_trainable:,d} / {n_params_total:,d}"
            f" ({100 * n_params_trainable / n_params_total:.2f}%)"
        )

        self.model = model
        self.tokenizer = tokenizer

        if self.debug:
            self.config.trainer.max_steps = 3
            self.config.trainer.eval_steps = 1
            self.config.trainer.logging_steps = 1
            self.config.trainer.save_strategy = "no"
            self.config.trainer.load_best_model_at_end = False

    def train(self, train_sentences, train_labels, val_sentences, val_labels, resume_from_checkpoint=None, **kwargs):
        # apply label mapping
        train_labels = apply_label_mapping(train_labels, self.config.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.config.label_mapping)

        # prepare the data
        train_dataset = self._prepare_dataset(train_sentences, train_labels, desc="train")
        eval_dataset = self._prepare_dataset(val_sentences, val_labels, desc="val")

        # filter train data
        if "difficulty_filter" in self.config.preprocessing:
            # load difficulty scores
            difficulty_scores = pd.read_csv(self.config.preprocessing.difficulty_filter.path, index_col=0)
            difficulty_scores = difficulty_scores[self.config.preprocessing.difficulty_filter.score_name]

            # split indices by difficulty
            train_difficult_idx, train_easy_idx = split_indices(
                train_sentences.index,
                difficulty_scores <= self.config.preprocessing.difficulty_filter.score_threshold,
                p=self.config.preprocessing.difficulty_filter.p,
            )
            n_difficult = len(train_difficult_idx)
            n_easy = len(train_easy_idx)

            # filter samples by difficulty
            mask = pd.Series(False, index=train_sentences.index)
            mask[train_difficult_idx] = True
            mask[train_easy_idx] = True
            train_indices = np.arange(0, len(train_dataset))[mask]
            n_total = len(train_dataset)
            n_filtered = len(train_indices)

            # print filter summary
            logger.info(
                f"Filtered train data based on difficulty: "
                f"{n_filtered:,d} / {n_total:,d} ({100 * n_filtered / n_total:.2f}%), "
                f"{n_difficult:,d} difficult, {n_easy:,d} easy"
            )
        else:
            train_indices = np.arange(0, len(train_dataset))

        # train the model
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer_config = WeightedLossTrainerConfig(
            output_dir=self.output_dir,
            report_to="wandb" if is_wandb_logging() else "none",
            **self.config.trainer,
        )
        self.trainer = WeightedLossTrainer(
            model=self.model,
            args=trainer_config,
            data_collator=data_collator,
            train_dataset=train_dataset.select(train_indices),
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # predict
        train_predictions = self.trainer.predict(train_dataset).predictions.argmax(axis=1)
        val_predictions = self.trainer.predict(eval_dataset).predictions.argmax(axis=1)

        # invert label mapping
        train_predictions = apply_inverse_label_mapping(train_predictions, self.config.label_mapping)
        val_predictions = apply_inverse_label_mapping(val_predictions, self.config.label_mapping)

        return train_predictions, val_predictions

    def predict(self, sentences):
        if self.trainer is None:
            raise ValueError("The model has not been trained yet. Please call the train() method first.")

        # prepare the data
        dataset = self._prepare_dataset(sentences, desc="predict")

        # predict
        predictions = self.trainer.predict(dataset).predictions.argmax(axis=1)

        # invert label mapping
        predictions = apply_inverse_label_mapping(predictions, self.config.label_mapping)

        return predictions

    def _preprocess(self, sample):
        return self.tokenizer(sample["sentence"], return_tensors="pt", **self.config.preprocessing.tokenizer)

    def _prepare_dataset(self, sentences, labels=None, desc=None):
        if labels is None:
            dataset = {"sentence": sentences}
        else:
            dataset = {"sentence": sentences, "label": labels}
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.map(
            self._preprocess,
            batched=self.config.preprocessing.batch_size,
            desc="Preprocessing" if desc is None else f"Preprocessing ({desc})",
        )
        return dataset

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # log to wandb if available
        if is_wandb_logging():
            cm = wandb.plot.confusion_matrix(
                probs=logits, y_true=labels, class_names=list(self.config.label_mapping.keys())
            )
            wandb.log({"confusion_matrix": cm})

        return {
            "accuracy": np.mean(predictions == labels),
            "score": 0.5 * (2 - np.abs(predictions - labels).mean()),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }
