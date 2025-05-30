"""This module provides the base class for all our pipelines."""

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
    ):
        """Initialize the base pipeline.

        Args:
            config (DictConfig | ListConfig): Configuration for the pipeline.
            device (str | torch.device | None, optional): Device to use for training and inference.
            output_dir (str, optional): Directory to save outputs. Defaults to "output".
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
        """
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.debug = debug

    def train(
        self,
        train_sentences: pd.Series,
        train_labels: pd.Series,
        val_sentences: pd.Series,
        val_labels: pd.Series,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the model and make predictions for the training and validation set.

        Args:
            train_sentences (pd.Series): Training sentences.
            train_labels (pd.Series): Labels for the training sentences.
            val_sentences (pd.Series): Validation sentences.
            val_labels (pd.Series): Labels for the validation sentences.

        Returns:
            tuple[np.ndarray, np.ndarray]: Predictions for the training and validation set.
        """
        raise NotImplementedError()

    def predict(self, sentences: pd.Series) -> np.ndarray:
        """Make predictions for the given sentences.

        Args:
            sentences (pd.Series): Sentences to make predictions for.

        Returns:
            predecitions(np.ndarray): Predictions for the given sentences.
        """
        raise NotImplementedError()
