import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig


class BasePipeline():
    """Base class for pipelines."""

    def __init__(self, config: DictConfig | ListConfig, device: str | torch.device | None = None):
        self.config = config
        self.device = device

    def train(self, train_sentences: pd.Series, train_labels: pd.Series, val_sentences: pd.Series, val_labels: pd.Series) -> np.ndarray:
        """Train the model and return predictions for the training set."""
        raise NotImplementedError("train() not implemented")

    def predict(self, sentences: pd.Series) -> np.ndarray:
        """Make predictions for the given sentences."""
        raise NotImplementedError("predict() not implemented")
