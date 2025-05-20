import logging

import numpy as np
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

from utils import apply_inverse_label_mapping, apply_label_mapping

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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.args.class_weights, device=model.device))
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
            **self.config.model, label2id=label2id, id2label=id2label
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
            model = get_peft_model(model, LoraConfig(**self.config.peft))
            logger.info(f"Using PEFT adapter: {self.config.peft}")

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

        # train the model
        # TODO use weighted loss to handle imbalanced classes
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
            train_dataset=train_dataset,
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
