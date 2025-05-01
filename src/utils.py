import logging
import os
import sys

import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

LABEL_MAPPING_REG = {"negative": -1, "neutral": 0, "positive": 1}
LABEL_MAPPING_CLA = {"negative": 0, "neutral": 1, "positive": 2}


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
    if verbose:
        logger.info(f"Loading the config file from '{config_path}'.")

    # load the config file
    config = OmegaConf.load(config_path)

    # overwrite the config
    if isinstance(overwrite, dict):
        overwrite = [f"{key}={value if value is not None else 'null'}" for key, value in overwrite.items()]
    config.merge_with_dotlist(overwrite)

    return config


def load_data(path, label_mapping=None):
    # load data
    dataset = pd.read_csv(path, index_col=0)
    # apply label mapping
    if label_mapping is not None:
        dataset["label"] = dataset["label"].map(label_mapping)

    return dataset


def evaluate_score(labels, predictions):
    return 0.5 * (2 - mean_absolute_error(labels, predictions))


def save_predictions(path, test_ids, test_predictions, label_mapping):
    submission = pd.DataFrame({"id": test_ids, "label": test_predictions})

    # revert label mapping
    label_mapping_rev = {value: key for key, value in label_mapping.items()}
    submission["label"] = submission["label"].map(label_mapping_rev)

    # save submission
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission.to_csv(path, index=False)
    logger.info(f"Submission saved to '{path}'.")
