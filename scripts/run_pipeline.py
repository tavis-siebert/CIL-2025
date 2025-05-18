import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "src").absolute()))

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from cache import CACHE
from pipelines import load_pipeline
from utils import (
    evaluate_score,
    get_config,
    get_device,
    load_data,
    save_predictions,
    setup_logging,
)

logger = logging.getLogger(__name__)


def main(args):
    # initialize cache
    CACHE.init(cache_dir=args.cache)

    # load config file
    config = get_config(args.config)

    # get device handler
    device = get_device(args.device, verbose=False)

    # load pipeline
    pipeline = load_pipeline(config.pipeline, device=device, output_dir=args.out, debug=args.debug)
    logger.info(f"Loaded pipeline: {config.pipeline.name}")

    # load train dataset and create splits
    train_dataset = load_data(Path(args.data) / "training.csv")
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_dataset["sentence"],
        train_dataset["label"],
        test_size=config.val_split_size,
        stratify=train_dataset["label"],
        random_state=config.seed,
    )

    if args.debug:
        train_sentences = train_sentences[:100]
        train_labels = train_labels[:100]
        val_sentences = val_sentences[:10]
        val_labels = val_labels[:10]

    # train and evaluate the model
    train_predictions, val_predictions = pipeline.train(train_sentences, train_labels, val_sentences, val_labels)
    score_train = evaluate_score(train_labels, train_predictions)
    score_val = evaluate_score(val_labels, val_predictions)
    logger.info(f"Score (training set): {score_train:.05f}")
    logger.info(f"Score (validation set): {score_val:.05f}")

    # load test dataset
    test_dataset = load_data(Path(args.data) / "test.csv")
    test_ids = test_dataset.index
    test_sentences = test_dataset["sentence"]

    if args.debug:
        test_ids = test_ids[:10]
        test_sentences = test_sentences[:10]

    # make predictions
    test_predictions = pipeline.predict(test_sentences)

    # save predictions
    save_predictions(args.out / "submission.csv", test_ids, test_predictions)


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
        "--cache",
        type=Path,
        default="output/cache",
        help="The path to the cache folder. (default: output/cache)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="output/submissions",
        help="The path to the output folder containing the submission file. (default: output/submissions)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="The device on which to compute. (default: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag whether to run in debug mode.",
    )
    args = parser.parse_args()

    main(args)
