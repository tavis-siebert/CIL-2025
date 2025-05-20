import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig


class BasePipeline:
    """Base class for pipelines."""

    def __init__(
        self,
        config: DictConfig | ListConfig,
        device: str | torch.device | None = None,
        output_dir: str = "output",
        debug: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.debug = debug
        self.verbose = verbose

    def train(
        self,
        train_sentences: pd.Series,
        train_labels: pd.Series,
        val_sentences: pd.Series,
        val_labels: pd.Series,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the model and make predictions for the training and validation set."""
        raise NotImplementedError()

    def predict(self, sentences: pd.Series) -> np.ndarray:
        """Make predictions for the given sentences."""
        raise NotImplementedError()
