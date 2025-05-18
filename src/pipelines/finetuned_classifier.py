import logging

import numpy as np
from datasets import Dataset
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import apply_inverse_label_mapping, apply_label_mapping

from .base import BasePipeline

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


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
            **self.config.model, label2id=label2id, id2label=id2label
        )
        model = model.to(self.device)
        logger.info(f"Loaded model: {self.config.model.pretrained_model_name_or_path}")
        logger.info(f"Using device: {model.device}")

        self.model = model
        self.tokenizer = tokenizer

        # TODO freeze model parameters
        # TODO use PEFT

        # print model summary
        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Number of trainable parameters: {n_params_trainable:,d} / {n_params_total:,d}"
            f" ({100 * n_params_trainable / n_params_total:.2f}%)"
        )

        if self.debug:
            self.config.trainer.max_steps = 3
            self.config.trainer.eval_steps = 1
            self.config.trainer.logging_steps = 1
            self.config.trainer.save_strategy = "no"
            self.config.trainer.load_best_model_at_end = False

    def train(self, train_sentences, train_labels, val_sentences, val_labels):
        # apply label mapping
        train_labels = apply_label_mapping(train_labels, self.config.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.config.label_mapping)

        # prepare the data
        train_dataset = self._prepare_dataset(train_sentences, train_labels, desc="train")
        eval_dataset = self._prepare_dataset(val_sentences, val_labels, desc="val")

        # train the model
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer_config = TrainingArguments(
            output_dir=self.output_dir,
            **self.config.trainer,
        )
        self.trainer = Trainer(
            model=self.model,
            args=trainer_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        self.trainer.train()

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
        if wandb is not None and wandb.run is not None:
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=logits, y_true=labels, class_names=list(self.config.label_mapping.keys()))})

        return {
            "accuracy": np.mean(predictions == labels),
            "score": 0.5 * (2 - np.abs(predictions - labels).mean()),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }
