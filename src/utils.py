import logging
import os
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, format=None, dateformat=None, other_loggers=[]):
    if format is None:
        format="[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)-5d %(message)s"
    if dateformat is None:
        dateformat="%Y-%m-%d %H:%M:%S"

    # configure logging system
    logging.basicConfig(
        stream=sys.stdout, # log to stdout instead of stderr to sync with print()
        level=level,
        format=format,
        datefmt=dateformat,
    )
    logging.captureWarnings(True)


def get_config(config_path, overwrite={}, verbose=True):
    # load the config file
    config = OmegaConf.load(config_path)

    # overwrite the config
    if isinstance(overwrite, dict):
        overwrite = [f"{key}={value if value is not None else 'null'}" for key, value in overwrite.items()]
    config.merge_with_dotlist(overwrite)

    if verbose:
        logger.info(f"Loaded config: {config_path}")
    return config


def get_device(device="auto", verbose=True):
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


def _check_valid_labels(labels):
    valid_labels = ["negative", "neutral", "positive"]
    if not pd.Series(labels).isin(valid_labels).all():
        raise ValueError(f"Labels contain invalid labels. Allowed labels are: {valid_labels}")


def load_data(path):
    dataset = pd.read_csv(path, index_col=0)
    return dataset


def apply_label_mapping(labels, label_mapping):
    labels = pd.Series(labels).map(label_mapping)
    return labels


def apply_inverse_label_mapping(labels, label_mapping):
    label_mapping_rev = {value: key for key, value in label_mapping.items()}
    labels = pd.Series(labels).map(label_mapping_rev)
    return labels


def evaluate_score(labels, predictions):
    _check_valid_labels(labels)
    _check_valid_labels(predictions)

    # convert labels to numeric labels
    label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
    labels = apply_label_mapping(labels, label_mapping)
    predictions = apply_label_mapping(predictions, label_mapping)

    # compute the score
    score = 0.5 * (2 - mean_absolute_error(labels, predictions))
    return score


def save_predictions(path, ids, predictions):
    _check_valid_labels(predictions)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission = pd.DataFrame({"id": ids, "label": predictions})
    submission.to_csv(path, index=False)
    logger.info(f"Submission saved to '{path}'.")
