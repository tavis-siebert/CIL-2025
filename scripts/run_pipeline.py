import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "src").absolute()))

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from pipelines import load_pipeline
from utils import (
    LABEL_MAPPING_CLA,
    LABEL_MAPPING_REG,
    evaluate_score,
    get_config,
    load_data,
    save_predictions,
    setup_logging,
)

logger = logging.getLogger(__name__)


def main(args):
    logger.info("### SETUP ###")

    # load config file
    config = get_config(args.config)

    # load pipeline
    pipeline = load_pipeline(config.pipeline)
    logger.info(f"Loaded pipeline '{config.pipeline.name}'.")

    if config.label_mapping == "regression":
        label_mapping = LABEL_MAPPING_REG
    elif config.label_mapping == "classification":
        label_mapping = LABEL_MAPPING_CLA
    else:
        raise ValueError(f"Unknown label mapping: {config.label_mapping}")

    logger.info("### TRAIN ###")

    # load train dataset and create splits
    train_dataset = load_data(Path(args.data) / "training.csv", label_mapping=label_mapping)
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_dataset["sentence"],
        train_dataset["label"],
        test_size=config.val_split_size,
        stratify=train_dataset["label"],
        random_state=config.seed,
    )

    # train the model
    train_predictions = pipeline.train(train_sentences, train_labels, val_sentences, val_labels)
    score_train = evaluate_score(train_labels, train_predictions)
    logger.info(f"Score (training set): {score_train:.05f}")

    # evaluate model
    val_predictions = pipeline.predict(val_sentences)
    score_val = evaluate_score(val_labels, val_predictions)
    logger.info(f"Score (validation set): {score_val:.05f}")


    logger.info("### TEST ###")

    # load test dataset
    test_dataset = load_data(Path(args.data) / "test.csv")
    test_ids = test_dataset.index
    test_sentences = test_dataset["sentence"]

    # make predictions
    test_predictions = pipeline.predict(test_sentences)

    # save predictions
    save_predictions(args.out, test_ids, test_predictions, label_mapping)


if __name__ == "__main__":
    # configure logging
    setup_logging()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="The path to the config file.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="The path to the folder containing 'training.csv' and 'test.csv'. (default: data)",
    )
    parser.add_argument(
        "--out",
        default="output/submissions/submission.csv",
        help="The path to the submission file. (default: output/submissions/submission.csv)",
    )
    args = parser.parse_args()

    main(args)
