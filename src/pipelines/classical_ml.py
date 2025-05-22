import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from preprocessing import apply_preprocessing
from utils import (
    apply_inverse_label_mapping,
    apply_label_mapping,
    load_glove_embeddings,
    sentences_to_glove_embeddings,
)

from .base import BasePipeline


class ClassicalMLPipeline(BasePipeline):
    """Classical machine learning pipeline for text classification."""

    def __init__(self, config, device=None, verbose=True):
        super().__init__(config, device=device, verbose=verbose)

        # configure label mapping
        if config.label_mapping == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        elif config.label_mapping == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        else:
            raise ValueError(f"Unknown label mapping: {config.label_mapping}")

        # configure vectorizer
        config_vectorizer = OmegaConf.to_container(config.vectorizer)
        if "ngram_range" in config_vectorizer:
            config_vectorizer["ngram_range"] = tuple(config_vectorizer["ngram_range"])
        self.vectorizer_type = config_vectorizer.pop("type")
        if self.vectorizer_type == "CountVectorizer":
            self.vectorizer = CountVectorizer(**config_vectorizer)
        elif self.vectorizer_type == "TfidfVectorizer":
            self.vectorizer = TfidfVectorizer(**config_vectorizer)
        elif self.vectorizer_type == "GloVe":
            glove_path = config_vectorizer.pop("path")
            self.glove_embeddings = load_glove_embeddings(glove_path)
        else:
            raise ValueError(f"Unknown vectorizer type: {config_vectorizer['type']}")

        # configure model
        config_model = OmegaConf.to_container(config.model)
        self.model_type = config_model.pop("type")
        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression(**config_model)
        elif self.model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**config_model)
        elif self.model_type == "GradientBoostingClassifier":
            self.model = GradientBoostingClassifier(**config_model)
        elif self.model_type == "XGBClassifier":
            self.model = XGBClassifier(**config_model)
        else:
            raise ValueError(f"Unknown model type: {config_model['type']}")

        # configure preprocessing
        self.preprocessing_rules = set(OmegaConf.to_container(config.preprocessing)) if "preprocessing" in config else None

    def train(self, train_sentences, train_labels, val_sentences, val_labels):
        # apply label mapping
        train_labels = apply_label_mapping(train_labels, self.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.label_mapping)

        # apply preprocessing for train
        if self.preprocessing_rules:
            train_sentences = apply_preprocessing(train_sentences, self.preprocessing_rules)

        # get embeddings for train
        if self.vectorizer_type == "GloVe":
            train_embeddings = sentences_to_glove_embeddings(train_sentences, self.glove_embeddings)
        else:
            train_embeddings = self.vectorizer.fit_transform(train_sentences)

        # train
        self.model.fit(train_embeddings, train_labels)

        # predict
        train_predictions = self.predict(train_sentences)
        val_predictions = self.predict(val_sentences)

        return train_predictions, val_predictions

    def predict(self, sentences):
        # apply preprocessing
        if self.preprocessing_rules:
            sentences = apply_preprocessing(sentences, self.preprocessing_rules)

        # get embeddings
        if self.vectorizer_type == "GloVe":
            embeddings = sentences_to_glove_embeddings(sentences, self.glove_embeddings)
        else:
            embeddings = self.vectorizer.transform(sentences)

        # predict
        predictions = self.model.predict(embeddings)

        # apply inverse label mapping
        predictions = pd.Series(predictions, index=sentences.index)
        predictions = apply_inverse_label_mapping(predictions, self.label_mapping)

        return predictions
