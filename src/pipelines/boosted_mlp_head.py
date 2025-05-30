"""Module for boosted MLP head pipeline.

Currently only supports MLPs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cache import load_embeddings
from utils import apply_inverse_label_mapping, apply_label_mapping

from .base import BasePipeline


class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        dropout_p=0.3,
        mode: str = "regression",
    ):
        """
        A simple feed-forward MLP.

        Args:
          hidden_sizes:  list/tuple of hidden layer sizes.
          dropout_p:     dropout probability.
          mode:          'regression' or 'classification'.
        """
        super().__init__()

        if mode == "classification":
            out_size = 3
        else:
            out_size = 1

        layers = []
        if hidden_sizes == []:
            layers.append(nn.LazyLinear(out_size))
        else:
            layers.append(nn.LazyLinear(hidden_sizes[0]))
            layers.append(nn.ReLU())

            for h0, h1 in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(nn.Linear(h0, h1))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))

            layers.append(nn.Linear(hidden_sizes[-1], out_size))

        # the actual model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BoostedMLPHeadModel(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        embeds_file = "embeddings.npz"
        if self.config.embed_pipeline == "huggingface":
            embeds_file = f"embeddings_{self.config.embed_type}.npz"
        self.embeddings = load_embeddings(self.config.embed_pipeline, self.config.embed_model, embeds_file)

        self.mode = self.config.mode
        if self.mode == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        else:
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

        # training and model
        self.boost_rate = self.config.boost_rate
        self.epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size

        self.learners = nn.ModuleList(
            [MLP(self.config.hidden_sizes, self.config.dropout_p, self.mode) for _ in range(self.config.n_learners)]
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # initialize the residuals to just the labels (subseqhent iterations make F_pred != 0)
        N = X.size(0)
        if self.mode == "regression":
            F_pred = torch.zeros_like(y.unsqueeze(1), device=self.device)
        else:
            F_pred = torch.zeros(N, 3, device=self.device)

        for h_i in tqdm(self.learners, desc="learners"):
            # compute pseudo-residuals
            loss_fn = nn.L1Loss()
            if self.mode == "regression":
                residual = (y.unsqueeze(1) - F_pred).detach()
            else:
                prob = F_pred.log_softmax(dim=1).exp()
                # gradient of CE wrt logits = prob - one_hot(y)
                one_hot = F.one_hot(y.long(), num_classes=3).float().to(self.device)
                residual = (one_hot - prob).detach()

            # train on (X, residual)
            optimizer = torch.optim.Adam(h_i.parameters(), lr=1e-3)
            dataset = torch.utils.data.TensorDataset(X, residual)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            h_i.to(self.device)

            h_i.train()
            for _ in tqdm(range(self.epochs), desc="epochs"):
                for x, res in loader:
                    optimizer.zero_grad()
                    pred = h_i(x)
                    loss = loss_fn(pred, res)
                    loss.backward()
                    optimizer.step()

            # update ensemble prediction:
            h_i.eval()
            with torch.no_grad():
                # not good practice but in this case the dataset is small enough
                F_pred += self.boost_rate * h_i(X)

    def train(
        self,
        train_sentences: pd.Series,
        train_labels: pd.Series,
        val_sentences: pd.Series,
        val_labels: pd.Series,
        **kwargs,
    ):
        embeddings = self.embeddings["train_embeddings"]

        train_embeddings = torch.from_numpy(embeddings[train_sentences.index]).float().to(self.device)
        val_embeddings = torch.from_numpy(embeddings[val_sentences.index]).float().to(self.device)

        train_labels = apply_label_mapping(train_labels, self.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.label_mapping)
        train_labels = torch.from_numpy(train_labels.values).float().to(self.device)
        val_labels = torch.from_numpy(val_labels.values).float().to(self.device)

        self.fit(train_embeddings, train_labels)

        train_predictions = self.preds_to_series(self.pred_tensor(train_embeddings), train_sentences.index)
        val_predictions = self.preds_to_series(self.pred_tensor(val_embeddings), val_sentences.index)

        return train_predictions, val_predictions

    @torch.no_grad()
    def pred_tensor(self, embeds: torch.Tensor):
        F_pred = None
        for h_i in self.learners:
            h_i.eval()
            out = h_i(embeds)
            if F_pred is None:
                F_pred = self.boost_rate * out
            else:
                F_pred += self.boost_rate * out

        if self.mode == "regression":
            return F_pred.squeeze().detach().cpu().numpy()
        return F_pred.argmax(dim=1).detach().cpu().numpy()

    def preds_to_series(self, preds, index):
        preds = pd.Series(preds, index=index)
        if self.mode == "regression":
            preds = preds.round().clip(-1, 1).astype(int)
        preds = apply_inverse_label_mapping(preds, self.label_mapping)
        return preds

    def predict(self, sentences: pd.Series) -> np.ndarray:
        embeddings = self.embeddings["test_embeddings"]
        test_embeddings = torch.from_numpy(embeddings[sentences.index]).float().to(self.device)
        preds = self.preds_to_series(self.pred_tensor(test_embeddings), sentences.index)
        return preds
