import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "src").absolute()))

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import load_data, setup_logging

logger = logging.getLogger(__name__)


def main(args):
    path = Path(args.out) / args.pipeline / args.model.replace("/", "__")
    if path.is_dir():
        raise ValueError(f"The embeddings already exist. Please delete the folder to proceed: {path}")

    # load datasets
    train_dataset = load_data(Path(args.data) / "training.csv")
    test_dataset = load_data(Path(args.data) / "test.csv")

    if args.pipeline == "sentencetransformer":
        # load pipeline
        model = SentenceTransformer(args.model)
        logger.info(f"Loaded SentenceTransformer pipeline with model: {args.model}")

        # generate embeddings
        train_embeddings = model.encode(train_dataset["sentence"], show_progress_bar=True, batch_size=args.batch_size)
        test_embeddings = model.encode(test_dataset["sentence"], show_progress_bar=True, batch_size=args.batch_size)

        # save embeddings
        path_embeddings = path / "embeddings.npz"
        np.savez_compressed(
            path_embeddings,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
        )
        logger.info(f"Saved embeddings: {path_embeddings}")
    elif args.pipeline == "huggingface":
        # load pipeline
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        logger.info(f"Loaded HuggingFace pipeline with model: {args.model}")

        # generate embeddings and predictions
        def encode(sentences, batch_size):
            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                sentences,
                batch_size=batch_size,
                collate_fn=lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512),
                shuffle=False,
            )

            # generate embeddings and predictions
            def postprocess_embeddings(embeddings, attention_mask):
                # apply mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                # normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                return embeddings

            embeddings_all = []
            predictions_all = []
            for batch in tqdm(dataloader):
                with torch.no_grad():
                    outputs = model(**batch, output_hidden_states=True)
                # extract embeddings from last hidden state
                embeddings = postprocess_embeddings(outputs.hidden_states[-1], batch["attention_mask"])
                embeddings = embeddings.detach().cpu().numpy()
                # extract predictions from logits
                predictions = torch.softmax(outputs.logits, dim=-1)
                predictions = predictions.detach().cpu().numpy()
                # append to lists
                embeddings_all.append(embeddings)
                predictions_all.append(predictions)
            embeddings_all = np.concatenate(embeddings_all, axis=0)
            predictions_all = np.concatenate(predictions_all, axis=0)
            predictions_all = pd.DataFrame(predictions_all, index=sentences.index, columns=model.config.id2label.values())

            return embeddings_all, predictions_all

        train_embeddings, train_predictions = encode(train_dataset["sentence"], args.batch_size)
        test_embeddings, test_predictions = encode(test_dataset["sentence"], args.batch_size)

        # save embeddings and predictions
        os.makedirs(path, exist_ok=True)
        path_embeddings = path / "embeddings.npz"
        path_predictions_train = path / "predictions_train.csv"
        path_predictions_test = path / "predictions_test.csv"
        np.savez_compressed(
            path_embeddings,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
        )
        train_predictions.to_csv(path_predictions_train)
        test_predictions.to_csv(path_predictions_test)
        logger.info(f"Saved embeddings: {path_embeddings}")
        logger.info(f"Saved predictions (train): {path_predictions_train}")
        logger.info(f"Saved predictions (test): {path_predictions_test}")
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")

if __name__ == "__main__":
    # configure logging
    setup_logging()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        default="st",
        choices=["sentencetransformer", "huggingface"],
        help="The type of the pipeline to use.",
    )
    parser.add_argument(
        "--model",
        help="The name of the model recognized by the pipeline.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="The path to the folder containing 'training.csv' and 'test.csv'. (default: data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="output/embeddings",
        help="The path to the embeddings folder. (default: output/embeddings)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size used to compute the embeddings. (default: 64)",
    )
    args = parser.parse_args()

    main(args)
