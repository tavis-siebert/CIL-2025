import argparse
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import load_data, setup_logging

logger = logging.getLogger(__name__)


def main(args):
    path = Path(args.out) / args.model / "embeddings.npz"

    # load datasets
    train_dataset = load_data(Path(args.data) / "training.csv")
    test_dataset = load_data(Path(args.data) / "test.csv")

    # load model
    model = SentenceTransformer(args.model)
    logger.info(f"Loaded model: {args.model}")

    # generate embeddings
    train_embeddings = model.encode(train_dataset["sentence"], show_progress_bar=True, batch_size=args.batch_size)
    test_embeddings = model.encode(test_dataset["sentence"].values, show_progress_bar=True, batch_size=args.batch_size)

    # save embeddings
    np.savez_compressed(
        path,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
    )
    logger.info(f"Saved embeddings to: {path}")


if __name__ == "__main__":
    # configure logging
    setup_logging()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="The name of the SentenceTransformer model.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="The path to the folder containing 'training.csv' and 'test.csv'. (default: data)",
    )
    parser.add_argument(
        "--out",
        default="output/embeddings",
        help="The path to the embeddings folder. (default: output/embeddings)",
    )
    args = parser.parse_args()

    main(args)
