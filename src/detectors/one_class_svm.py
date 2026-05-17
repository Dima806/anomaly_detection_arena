import numpy as np
from sklearn.svm import OneClassSVM

from src.detectors.base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    name = "OneClassSVM"

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        max_samples: int = 10000,
        random_state: int = 42,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.max_samples = max_samples
        self.random_state = random_state
        self._model: OneClassSVM | None = None

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        # Subsample training data to max_samples rows for speed
        rng = np.random.RandomState(self.random_state)
        if len(X) > self.max_samples:
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X = X[idx]
        self._model = OneClassSVM(kernel=self.kernel, nu=self.nu)
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return -self._model.score_samples(X)
