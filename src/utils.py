import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, format: str | None = None, dateformat: str | None = None):
    if format is None:
        format = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)-5d %(message)s"
    if dateformat is None:
        dateformat = "%Y-%m-%d %H:%M:%S"

    # configure logging system
    logging.basicConfig(
        stream=sys.stdout,  # log to stdout instead of stderr to sync with print()
        level=level,
        format=format,
        datefmt=dateformat,
    )
    logging.captureWarnings(True)


def get_config(
    config_path: str | Path,
    overwrite: dict[str, Any] = {},
    verbose: bool = True,
) -> DictConfig | ListConfig:
    # load the config file
    config = OmegaConf.load(config_path)

    # overwrite the config
    if isinstance(overwrite, dict):
        overwrite_dotlist = [f"{key}={value if value is not None else 'null'}" for key, value in overwrite.items()]
    config.merge_with_dotlist(overwrite_dotlist)

    if verbose:
        logger.info(f"Loaded config: {config_path}")
    return config


def get_device(device: str | torch.device = "auto", verbose: bool = True) -> torch.device:
    # create the device handler
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    if verbose:
        logger.info(f"Using device: {device}")
    return device


def ensure_reproducibility(seed: int | None = None, deterministic: bool = False, verbose: bool = True):
    """Set seeds and ensures usage of deterministic algorithms.

    Args:
        seed (int, optional): The seed set for each dataloader worker. Defaults
            to None.
        deterministic (bool, optional): Flag whether algorithms should be as
            deterministic as possible. Defaults to False.
        verbose (bool, optional): Flag whether to be verbose. Defaults to True.

    References:
        [1] https://pytorch.org/docs/stable/notes/randomness.html
    """
    # seed random number generators
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if verbose:
            logger.info(f"Set seeds: {seed}")

    # use deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if verbose:
            logger.info("Enabled deterministic algorithms.")


def _check_valid_labels(labels):
    valid_labels = ["negative", "neutral", "positive"]
    if not pd.Series(labels).isin(valid_labels).all():
        raise ValueError(f"Labels contain invalid labels. Allowed labels are: {valid_labels}")


def load_data(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(path, index_col=0)
    return dataset


def apply_label_mapping(labels, label_mapping: dict[str, Any]) -> pd.Series:
    labels = pd.Series(labels).map(label_mapping)
    return labels


def apply_inverse_label_mapping(labels, label_mapping: dict[str, Any]) -> pd.Series:
    label_mapping_rev = {value: key for key, value in label_mapping.items()}
    labels = pd.Series(labels).map(label_mapping_rev)
    return labels


def split_indices(indices: pd.Index, condition: pd.Series, p: float = 1) -> tuple[pd.Index, pd.Index]:
    # filter indices by condition
    indices_a = indices.intersection(condition.index[condition])
    indices_b = indices.intersection(condition.index[~condition])

    # compute number of samples per group
    n_samples_a = len(indices_a) if p == 1 else min(len(indices_a), len(indices_b) * p / (1-p))
    n_samples_b = len(indices_b) if p == 0 else n_samples_a * (1-p) / p
    n_samples_a = int(n_samples_a)
    n_samples_b = int(n_samples_b)

    # sample indices
    indices_a = pd.Index(indices_a.to_series().sample(n=n_samples_a, random_state=0))
    indices_b = pd.Index(indices_b.to_series().sample(n=n_samples_b, random_state=1))

    return indices_a, indices_b


def evaluate_score(labels, predictions) -> float:
    _check_valid_labels(labels)
    _check_valid_labels(predictions)

    # convert labels to numeric labels
    label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
    labels = apply_label_mapping(labels, label_mapping)
    predictions = apply_label_mapping(predictions, label_mapping)

    # compute the score
    score = 0.5 * (2 - mean_absolute_error(labels, predictions))
    return float(score)


def save_predictions(path: str | Path, ids, predictions):
    _check_valid_labels(predictions)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission = pd.DataFrame({"id": ids, "label": predictions})
    submission.to_csv(path, index=False)
    logger.info(f"Submission saved to '{path}'.")


def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index


def sentences_to_glove_embeddings(sentences, glove_embeddings):
    dim = len(glove_embeddings[next(iter(glove_embeddings))])
    all_embeddings = []
    for sentence in sentences:
        words = sentence.split()
        embeddings = np.zeros((len(words), dim))
        for i, word in enumerate(words):
            if word in glove_embeddings:
                embeddings[i] = glove_embeddings[word]
            else:
                embeddings[i] = np.zeros(dim)
        all_embeddings.append(embeddings.mean(axis=0) if len(words) > 0 else np.zeros(dim))

    return pd.DataFrame(all_embeddings)
