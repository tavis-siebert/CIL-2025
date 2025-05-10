import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import apply_inverse_label_mapping, apply_label_mapping

from .base import BasePipeline


class BaselineBowLogreg(BasePipeline):
    """Bag-of-words and logistic regression baseline taken from the provided notebook."""

    def __init__(self, config, device=None):
        self.config = config

        # configure label mapping
        if config.label_mapping == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        elif config.label_mapping == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        else:
            raise ValueError(f"Unknown label mapping: {config.label_mapping}")

        # configure bag-of-words
        config_bow = OmegaConf.to_container(config.bow)
        if "ngram_range" in config_bow:
            config_bow["ngram_range"] = tuple(config_bow["ngram_range"])
        self.vectorizer = CountVectorizer(**config_bow)

        # configure logistic regression
        self.model = LogisticRegression(**config.logreg)

    def train(self, train_sentences, train_labels, val_sentences, val_labels):
        # apply label mapping
        train_labels = apply_label_mapping(train_labels, self.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.label_mapping)

        # train
        train_embeddings = self.vectorizer.fit_transform(train_sentences)
        self.model.fit(train_embeddings, train_labels)

        # predict
        train_predictions = self.predict(train_sentences)
        val_predictions = self.predict(val_sentences)

        return train_predictions, val_predictions

    def predict(self, sentences):
        # predict on sentences
        embeddings = self.vectorizer.transform(sentences)
        predictions = self.model.predict(embeddings)

        # apply inverse label mapping
        predictions = pd.Series(predictions, index=sentences.index)
        predictions = apply_inverse_label_mapping(predictions, self.label_mapping)

        return predictions
