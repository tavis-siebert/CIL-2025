import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from cache import load_embeddings
from utils import apply_label_mapping, apply_inverse_label_mapping
from .base import BasePipeline

class BaseClassifier(BasePipeline):
    """
    Implements a linear head over 
    """
    def __init__(
        self,
        config,
        device
    ):
        super().__init__(config, device)  # initialize self.config, self.device
        
        embeds_file = f"embeddings_{self.config.embed_type}.npz"
        self.embeddings = load_embeddings(self.config.embed_pipeline, self.config.embed_model, embeds_file)

        # model
        if config.mode == "regression":
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
            out_size = 1
        elif config.mode == "classification":
            self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
            out_size = 3
        else:
            raise ValueError(f"Unknown label mapping: {config.mode}")

        self.classifier = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_size)
        ).to(self.device)
    

    def train(self, train_sentences, train_labels, val_sentences, val_labels):

        embeddings = self.embeddings['train_embeddings']

        train_embeddings = torch.from_numpy(embeddings[train_sentences.index]).float().to(self.device)
        val_embeddings = torch.from_numpy(embeddings[val_sentences.index]).float().to(self.device)

        train_labels = apply_label_mapping(train_labels, self.label_mapping)
        val_labels = apply_label_mapping(val_labels, self.label_mapping)
        if self.config.mode == 'classification':
            train_labels = torch.from_numpy(train_labels.values).long().to(self.device)
            val_labels   = torch.from_numpy(val_labels.values).long().to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            train_labels = torch.from_numpy(train_labels.values).float().unsqueeze(1).to(self.device)
            val_labels   = torch.from_numpy(val_labels.values).float().unsqueeze(1).to(self.device)
            criterion = nn.L1Loss()

        train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() if self.config.mode == 'classification' else nn.L1Loss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4, weight_decay=0.01)

        self.classifier.train()
        for epoch in range(self.config.num_epochs):
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                pred = self.classifier(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        train_predictions = self.preds_to_series(self.predict_tensor(train_embeddings), train_sentences.index)
        val_predictions   = self.preds_to_series(self.predict_tensor(val_embeddings), val_sentences.index)
            
        return train_predictions, val_predictions

    def preds_to_series(self, preds, index):
        preds = pd.Series(preds, index=index)
        preds = apply_inverse_label_mapping(preds, self.label_mapping)
        return preds

    @torch.no_grad()
    def predict_tensor(self, embeds):
        self.classifier.eval()
        preds = self.classifier(embeds)
        if self.config.mode == 'classification':
            return preds.argmax(dim=1).detach().cpu().numpy()
        return preds.squeeze().detach().cpu().numpy()

    def predict(self, sentences):
        embeddings = self.embeddings['test_embeddings']
        test_embeddings = torch.from_numpy(embeddings[sentences.index]).float().to(self.device)
        preds = self.preds_to_series(self.predict_tensor(test_embeddings), sentences.index)
        return preds