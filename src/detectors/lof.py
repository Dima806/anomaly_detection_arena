import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.detectors.base import BaseDetector


class LOFDetector(BaseDetector):
    name = "LOF"

    def __init__(self, n_neighbors: int = 20, n_jobs: int = 2) -> None:
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self._model: LocalOutlierFactor | None = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        # LOF is also transductive. Use novelty=True for inductive mode
        # (fit then predict on new data).
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            novelty=True,
        )
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return -self._model.score_samples(X)
