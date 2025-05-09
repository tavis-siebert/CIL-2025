from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .base import BasePipeline


class BaselineBowLogreg(BasePipeline):
    """Bag-of-words and logistic regression baseline taken from the provided notebook."""

    def __init__(self, config, device=None):
        self.config = config

        config_bow = OmegaConf.to_container(config.bow)
        if "ngram_range" in config_bow:
            config_bow["ngram_range"] = tuple(config_bow["ngram_range"])
        self.vectorizer = CountVectorizer(**config_bow)

        self.model = LogisticRegression(**config.logreg)

    def train(self, train_sentences, train_labels, val_sentences, val_labels):
        # train
        train_embeddings = self.vectorizer.fit_transform(train_sentences)
        self.model.fit(train_embeddings, train_labels)

        # make predictions for training data
        train_predictions = self.model.predict(train_embeddings)
        return train_predictions

    def predict(self, sentences):
        embeddings = self.vectorizer.transform(sentences)
        predictions = self.model.predict(embeddings)
        return predictions
