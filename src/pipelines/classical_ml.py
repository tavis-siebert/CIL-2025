"""Module for classical ML pipeline that supports several sklearn models."""

import logging

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import OmegaConf
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from preprocessing import apply_preprocessing
from utils import (
    apply_inverse_label_mapping,
    apply_label_mapping,
    load_glove_embeddings,
    sentences_to_glove_embeddings,
)

from .base import BasePipeline

logger = logging.getLogger(__name__)


def create_model(model_config, label_mapping):
    """Create a sklearn model based on the provided configuration.

    Args:
        model_config (dict): Configuration dictionary for the model.
        label_mapping (str): Type of label mapping, either "regression" or "classification".

    Returns:
        model: An instance of the specified machine learning model.
    """
    model_type = model_config.pop("type")

    if label_mapping == "regression":
        match model_type:
            case "XGBRegressor":
                return XGBRegressor(**model_config)
            case _:
                raise ValueError("Unknown model type for regression: {model_type}")

    if label_mapping == "classification":
        # classification models
        match model_type:
            case "LogisticRegression":
                return LogisticRegression(**model_config)
            case "RandomForestClassifier":
                return RandomForestClassifier(**model_config)
            case "GradientBoostingClassifier":
                return GradientBoostingClassifier(**model_config)
            case "XGBClassifier":
                return XGBClassifier(**model_config)
            case "SVC":
                return SVC(**model_config)
            case "OneVsRestClassifier":
                return OneVsRestClassifier(
                    create_model(model_config.pop("estimator"), label_mapping),
                    **model_config,
                )
            case "StackingClassifier":
                if type(model_config["estimators"]) is list:
                    estimators = [
                        create_model(estimator_config, label_mapping)
                        for estimator_config in enumerate(
                            model_config.pop("estimators")
                        )
                    ]
                elif type(model_config["estimators"]) is dict:
                    estimators = [
                        (name, create_model(estimator_config, label_mapping))
                        for name, estimator_config in model_config.pop(
                            "estimators"
                        ).items()
                    ]

                if "final_estimator" in model_config:
                    final_estimator = create_model(
                        model_config.pop("final_estimator"), label_mapping
                    )
                else:
                    final_estimator = None

                return StackingClassifier(
                    estimators, final_estimator=final_estimator, **model_config
                )
            case _:
                raise ValueError(
                    f"Unknown model type '{model_type}' for classification."
                )

    raise ValueError(f"Unknown label mapping: {label_mapping}")


class ClassicalMLPipeline(BasePipeline):
    """Classical machine learning pipeline for several sklearn models.

    This pipeline supports various vectorizers, models, and preprocessing steps.
    It can be used for both regression and classification tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # configure label mapping
        if self.config.label_mapping == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        elif self.config.label_mapping == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        else:
            raise ValueError(f"Unknown label mapping: {self.config.label_mapping}")

        # configure vectorizer
        vectorizer_config = OmegaConf.to_container(self.config.vectorizer)
        if "ngram_range" in vectorizer_config:
            vectorizer_config["ngram_range"] = tuple(vectorizer_config["ngram_range"])
        self.vectorizer_type = vectorizer_config.pop("type")
        if self.vectorizer_type == "CountVectorizer":
            self.vectorizer = CountVectorizer(**vectorizer_config)
        elif self.vectorizer_type == "TfidfVectorizer":
            self.vectorizer = TfidfVectorizer(**vectorizer_config)
        elif self.vectorizer_type == "GloVe":
            glove_path = vectorizer_config.pop("path")
            self.glove_embeddings = load_glove_embeddings(glove_path)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_config['type']}")

        # configure model
        model_config = OmegaConf.to_container(self.config.model)
        self.model = create_model(model_config, self.config.label_mapping)
        logger.info(f"Loaded model: {self.config.model.type}")

        # configure preprocessing
        self.preprocessing_rules = (
            set(OmegaConf.to_container(self.config.preprocessing))
            if "preprocessing" in self.config
            else None
        )

        self.use_undersampling = self.config.get("undersampling", False)
        self.use_oversampling = self.config.get("oversampling", False)
        if self.use_undersampling and self.use_oversampling:
            raise ValueError(
                "Cannot use both undersampling and oversampling at the same time."
            )

    def train(self, train_sentences, train_labels, val_sentences, val_labels, **kwargs):
        # apply label mapping
        if self.label_mapping:
            train_labels = apply_label_mapping(train_labels, self.label_mapping)
            val_labels = apply_label_mapping(val_labels, self.label_mapping)

        train_sentences_for_fit = train_sentences.copy()

        # apply preprocessing for train
        if self.preprocessing_rules:
            train_sentences_for_fit = apply_preprocessing(
                train_sentences, self.preprocessing_rules
            )

        # reduce train set size if specified
        if "percent_train_samples" in self.config:
            logger.warning(
                f"Warning: Reducing train set size to {self.config.percent_train_samples * 100}% ({len(train_sentences)} samples)"
            )
            train_sentences_for_fit = train_sentences[
                : int(len(train_sentences) * self.config.percent_train_samples)
            ]
            train_labels_for_fit = train_labels[
                : int(len(train_sentences) * self.config.percent_train_samples)
            ]
        else:
            train_sentences_for_fit = train_sentences_for_fit
            train_labels_for_fit = train_labels

        # get embeddings for train
        if self.vectorizer_type == "GloVe":
            train_embeddings = sentences_to_glove_embeddings(
                train_sentences_for_fit, self.glove_embeddings
            )
        else:
            train_embeddings = self.vectorizer.fit_transform(train_sentences_for_fit)

        if self.use_undersampling:
            undersampler = RandomUnderSampler()
            train_embeddings, train_labels_for_fit = undersampler.fit_resample(
                train_embeddings, train_labels_for_fit
            )
            logger.info(
                f"Undersampled train set to {len(train_labels_for_fit)} samples"
            )

        if self.use_oversampling:
            oversampler = RandomOverSampler()
            train_embeddings, train_labels_for_fit = oversampler.fit_resample(
                train_embeddings, train_labels_for_fit
            )
            logger.info(f"Oversampled train set to {len(train_labels_for_fit)} samples")

        # train
        self.model.fit(train_embeddings, train_labels_for_fit)

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
        if self.config.label_mapping == "classification":
            predictions = apply_inverse_label_mapping(predictions, self.label_mapping)

        return predictions
