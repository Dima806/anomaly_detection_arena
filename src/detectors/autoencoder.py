import numpy as np
import torch
import torch.nn as nn

from src.detectors.base import BaseDetector


class _AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseDetector):
    name = "Autoencoder"

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self._model: _AE | None = None

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(self.random_state)
        n, d = X.shape
        self._model = _AE(d, self.hidden_dim)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        Xt = torch.tensor(X, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(Xt)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                opt.zero_grad()
                nn.functional.mse_loss(self._model(batch), batch).backward()
                opt.step()
        self._model.eval()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        Xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = self._model(Xt)
        return ((Xt - recon) ** 2).mean(dim=1).numpy()
