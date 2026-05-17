import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.detectors.base import BaseDetector


class DBSCANDetector(BaseDetector):
    name = "DBSCAN"

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._nn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)
        core_mask = labels != -1
        # Fall back to all points if no core points were found
        reference = X[core_mask] if core_mask.sum() > 0 else X
        self._nn = NearestNeighbors(n_neighbors=1).fit(reference)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._nn is not None
        distances, _ = self._nn.kneighbors(X)
        return distances.ravel()
