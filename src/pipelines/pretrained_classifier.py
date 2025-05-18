import string

import pandas as pd

from cache import load_embeddings

from .base import BasePipeline


def preprocess_data(text: str, model_name: str) -> str:
    """Apply preprocessing of pre-trained models to the text."""

    if model_name in [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    ]:
        # reference: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    elif model_name in [
        "helinivan/english-sarcasm-detector",
        "helinivan/multilingual-sarcasm-detector",
    ]:
        # reference: https://huggingface.co/helinivan/english-sarcasm-detector
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    else:
        return text


def map_to_labels(predictions, model_name):
    """Convert predicted class probabilities to labels."""
    if model_name in [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    ]:
        return predictions.idxmax(axis=1)
    elif model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
        predictions = pd.DataFrame({
            "negative": predictions["1 star"] + predictions["2 stars"] / 2,
            "neutral": predictions["3 stars"] + predictions["2 stars"] / 2 + predictions["4 stars"] / 2,
            "positive": predictions["5 stars"] + predictions["4 stars"] / 2,
        })
        return predictions.idxmax(axis=1)
    elif model_name == "siebert/sentiment-roberta-large-english":
        return pd.cut(predictions["POSITIVE"], bins=[0, 0.33, 0.66, 1.0], labels=["negative", "neutral", "positive"])
    elif model_name == "tabularisai/multilingual-sentiment-analysis":
        predictions = pd.DataFrame({
            "negative": predictions["Very Negative"] + predictions["Negative"] / 2,
            "neutral": predictions["Neutral"] + predictions["Negative"] / 2 + predictions["Positive"] / 2,
            "positive": predictions["Very Positive"] + predictions["Very Positive"] / 2,
        })
        return predictions.idxmax(axis=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class PretrainedClassifier(BasePipeline):
    """Pretrained sentiment classifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # load predictions from embeddings cache
        self.predictions_train = load_embeddings("huggingface", self.config.model, "predictions_train.csv", load_kwargs={"index_col": 0})
        self.predictions_test = load_embeddings("huggingface", self.config.model, "predictions_test.csv", load_kwargs={"index_col": 0})

    def train(self, train_sentences, train_labels, val_sentences, val_labels):
        predictions_train = self.predictions_train.iloc[train_sentences.index]
        predictions_val = self.predictions_train.iloc[val_sentences.index]

        # convert predictions to labels
        predictions_train = map_to_labels(predictions_train, self.config.model)
        predictions_val = map_to_labels(predictions_val, self.config.model)

        return predictions_train, predictions_val

    def predict(self, sentences):
        predictions_test = self.predictions_test.iloc[sentences.index]

        # convert predictions to labels
        predictions_test = map_to_labels(predictions_test, self.config.model)

        return predictions_test
