import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "src").absolute()))

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cache import CACHE, get_embeddings_folder, save_embeddings
from pipelines.pretrained_classifier import preprocess_data
from utils import get_device, load_data, setup_logging

logger = logging.getLogger(__name__)


def main(args):
    # initialize cache
    CACHE.init(cache_dir=args.cache)
    embeddings_folder = get_embeddings_folder(args.pipeline, args.model)
    if CACHE.get_path(embeddings_folder).is_dir():
        raise FileExistsError(f"The embeddings already exist in the cache. Please delete the folder: {embeddings_folder}")

    # get device handler
    device = get_device(args.device, verbose=False)

    # load datasets
    train_dataset = load_data(Path(args.data) / "training.csv")
    test_dataset = load_data(Path(args.data) / "test.csv")

    if args.pipeline == "sentencetransformer":
        # load pipeline
        model = SentenceTransformer(args.model, device=device)
        logger.info(f"Loaded SentenceTransformer pipeline with model: {args.model}")
        logger.info(f"Using device: {model.device}")

        # generate embeddings
        train_embeddings = model.encode(train_dataset["sentence"], show_progress_bar=True, batch_size=args.batch_size)
        test_embeddings = model.encode(test_dataset["sentence"], show_progress_bar=True, batch_size=args.batch_size)

        # save embeddings
        embeddings = {
            "train_embeddings": train_embeddings,
            "test_embeddings": test_embeddings,
        }
        save_embeddings(embeddings, args.pipeline, args.model)
    elif args.pipeline == "huggingface":
        # load pipeline
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model = model.to(device)
        logger.info(f"Loaded HuggingFace pipeline with model: {args.model}")
        logger.info(f"Using device: {model.device}")

        # generate embeddings and predictions
        def encode(sentences, batch_size):
            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                sentences,
                batch_size=batch_size,
                collate_fn=lambda sentences: tokenizer([preprocess_data(s, args.model) for s in sentences], return_tensors="pt", padding=True, truncation=True, max_length=512),
                shuffle=False,
            )

            # generate embeddings and predictions
            def extract_embeddings(last_hidden_state, attention_mask):
                # reference: https://github.com/UKPLab/sentence-transformers/blob/68a61ec8e8f0497e5cddc0bc59f92b82ef62b54d/sentence_transformers/models/Pooling.py#L135
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # extract CLS token embedding
                embeddings_cls = last_hidden_state[:, 0]
                # apply mean pooling
                embeddings_mean = torch.sum(last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                # apply max pooling
                embeddings_max = torch.max(torch.where(input_mask_expanded == 0, -1e9, last_hidden_state), dim=1).values
                return embeddings_cls, embeddings_mean, embeddings_max

            embeddings_all_cls = []
            embeddings_all_mean = []
            embeddings_all_max = []
            predictions_all = []
            for batch in tqdm(dataloader):
                batch = batch.to(model.device)
                with torch.no_grad():
                    outputs = model(**batch, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1].detach()
                    logits = outputs.logits.detach()
                # extract embeddings from last hidden state
                embeddings = extract_embeddings(last_hidden_state, batch["attention_mask"])
                embeddings_all_cls.append(embeddings[0].cpu().numpy())
                embeddings_all_mean.append(embeddings[1].cpu().numpy())
                embeddings_all_max.append(embeddings[2].cpu().numpy())
                # extract predictions from logits
                predictions = torch.softmax(logits, dim=-1)
                predictions_all.append(predictions.cpu().numpy())
            embeddings_all = {
                "cls": np.concatenate(embeddings_all_cls, axis=0),
                "mean": np.concatenate(embeddings_all_mean, axis=0),
                "max": np.concatenate(embeddings_all_max, axis=0),
            }
            predictions_all = np.concatenate(predictions_all, axis=0)
            predictions_all = pd.DataFrame(predictions_all, index=sentences.index, columns=model.config.id2label.values())

            return embeddings_all, predictions_all

        train_embeddings, train_predictions = encode(train_dataset["sentence"], args.batch_size)
        test_embeddings, test_predictions = encode(test_dataset["sentence"], args.batch_size)

        # save embeddings
        for mode in ["cls", "mean", "max"]:
            embeddings = {
                "train_embeddings": train_embeddings[mode],
                "test_embeddings": test_embeddings[mode],
            }
            save_embeddings(embeddings, args.pipeline, args.model, f"embeddings_{mode}.npz")
        # save predictions to the embeddings cache
        save_embeddings(train_predictions, args.pipeline, args.model, "predictions_train.csv")
        save_embeddings(test_predictions, args.pipeline, args.model, "predictions_test.csv")
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")

if __name__ == "__main__":
    # configure logging
    setup_logging()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        default="sentencetransformer",
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
        "--cache",
        type=Path,
        default="output/cache",
        help="The path to the cache folder. (default: output/cache)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="The device on which to compute. (default: auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size used to compute the embeddings. (default: 64)",
    )
    args = parser.parse_args()

    main(args)
