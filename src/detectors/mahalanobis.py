import numpy as np

from src.detectors.base import BaseDetector


class MahalanobisDetector(BaseDetector):
    name = "Mahalanobis"

    def __init__(self) -> None:
        self._mu: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        self._mu = X.mean(axis=0)
        # np.cov returns a scalar for single-feature data; ensure at least 2-d
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        self._cov_inv = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._mu is not None and self._cov_inv is not None
        diff = X - self._mu
        return np.sqrt(np.maximum(np.sum(diff @ self._cov_inv * diff, axis=1), 0.0))
