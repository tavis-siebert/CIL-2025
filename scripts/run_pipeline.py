import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "src").absolute()))

import argparse
import contextlib
import logging
from pathlib import Path

from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from cache import CACHE
from pipelines import load_pipeline
from utils import (
    ensure_reproducibility,
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

    # initialize context
    if args.report_to is not None:
        import wandb

        kwargs = {
            "entity": args.report_to[0],
            "project": args.report_to[1],
            "config": OmegaConf.to_object(config),
        }
        if args.resume_wandb is not None:
            kwargs["resume"] = "must"
            kwargs["id"] = args.resume_wandb
        context_wandb = wandb.init(**kwargs)
    else:
        context_wandb = contextlib.nullcontext()

    with context_wandb:
        # set seeds and use deterministic algorithms
        ensure_reproducibility(seed=config.seed, deterministic=config.deterministic)

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
        train_predictions, val_predictions = pipeline.train(
            train_sentences,
            train_labels,
            val_sentences,
            val_labels,
            resume_from_checkpoint=args.resume,
        )
        score_train = evaluate_score(train_labels, train_predictions)
        score_val = evaluate_score(val_labels, val_predictions)
        cm_train = confusion_matrix(train_labels, train_predictions)
        cm_val = confusion_matrix(train_labels, train_predictions)
        logger.info(f"Score (train): {score_train:.05f}")
        logger.info(f"Score (val): {score_val:.05f}")
        logger.info(f"Confusion matrix (train):\n{cm_train}")
        logger.info(f"Confusion matrix (val):\n{cm_val}")
        if args.report_to is not None:
            wandb.log({"train_score": score_train, "val_score": score_val})

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
        default="output",
        help="The path to the output folder containing the submission file. (default: output/submissions)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="The device on which to compute. (default: auto)",
    )
    parser.add_argument(
        "--report-to",
        nargs=2,
        default=None,
        metavar=("ENTITY", "PROJECT"),
        help="The W&B target entity and project name. (default: None)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Flag whether to resume training from the last checkpoint. "
            "Optionally, a local path can be provided. (default: False)",
        ),
    )
    parser.add_argument(
        "--resume-wandb",
        default=None,
        help="The W&B run id which should be resumed (default: None).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag whether to run in debug mode.",
    )
    args = parser.parse_args()

    main(args)
