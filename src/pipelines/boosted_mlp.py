"""
Currently only supports MLPs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        dropout_p=0.3,
        mode: str='regression',
    ):
        """
        A simple feed-forward MLP.
        
        Args:
          hidden_sizes:  list/tuple of hidden layer sizes.
          dropout_p:     dropout probability.
          mode:          'regression' or 'classification'.
        """
        super().__init__() 

        layers = []

        layers.append(nn.LazyLinear(hidden_sizes[0]))
        layers.append(nn.ReLU())

        for h0, h1 in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(h0, h1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

        # final layer
        if mode == 'classification':
            out_size = 3
        elif mode == 'regression':
            out_size = 1
        else:
            raise ValueError(f"mode must be 'regression' or 'classification', got {mode!r}")

        layers.append(nn.Linear(hidden_sizes[-1], out_size))

        # the actual model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BoostedMLP:
    def __init__(
        self, 
        config
    ): 
        self.boost_rate = config.boost_rate
        self.epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = config.mode

        self.learners = nn.ModuleList(
            [MLP(config.hidden_sizes, config.dropout_p, config.mode) for _ in range(config.n_learners)]
        )
    
    def fit(self, X, y):
        # initialize the residuals to just the labels (subseqhent iterations make F_pred != 0)
        N = X.size(0)
        if self.mode == 'regression':
            F_pred = torch.zeros_like(y.unsqueeze(1), device=self.device)
        else:
            F_pred = torch.zeros(N, 3, device=self.device)

        for h_i in tqdm(self.learners, desc='learners'):
            # compute pseudo-residuals
            loss_fn = nn.L1Loss()
            if self.mode == 'regression':
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
            for _ in tqdm(range(self.epochs), desc='epochs'):
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

    def train(self, X, y, X_val, y_val):
        X, y = X.to(self.device), y.to(self.device)
        self.fit(X, y)
        train_preddictions = self.predict(X)
        return train_preddictions

    @torch.no_grad()
    def predict(self, X):
        X = X.to(self.device)
        F_pred = None
        for h_i in (self.learners):
            out = h_i(X)
            if F_pred is None:
                F_pred = self.boost_rate * out
            else:
                F_pred += self.boost_rate * out

        if self.mode == 'regression':
            return F_pred.squeeze().detach().cpu()

        return F_pred.argmax(dim=1).detach().cpu()