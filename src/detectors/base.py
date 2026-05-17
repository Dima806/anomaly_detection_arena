from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    name: str  # class-level attribute, human-readable

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector": ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Float64 array shape (n,). Higher = more anomalous."""

    def predict(self, X: np.ndarray, contamination: float) -> np.ndarray:
        """Return int array (n,): 1=anomaly, 0=normal. Uses top-contamination quantile."""
        scores = self.score(X)
        threshold = np.quantile(scores, 1.0 - contamination)
        return (scores >= threshold).astype(int)
