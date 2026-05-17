import numpy as np
from sklearn.ensemble import IsolationForest

from src.detectors.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "IsolationForest"

    def __init__(
        self,
        n_estimators: int = 100,
        n_jobs: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # sklearn IsolationForest.score_samples returns negative scores (lower = more anomalous)
        # Negate so higher = more anomalous
        assert self._model is not None
        return -self._model.score_samples(X)
