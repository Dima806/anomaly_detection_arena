import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.evaluation.metrics import precision_recall_f1_at_contamination


def sweep_contamination(
    detector: BaseDetector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep contamination rate and record precision/recall/f1 at each value.

    Parameters
    ----------
    detector:
        Detector instance (will be fit on X_train).
    X_train:
        Training features.
    X_test:
        Test features.
    y_test:
        True labels for X_test (1=anomaly, 0=normal).
    contamination_range:
        Array of contamination fractions to evaluate. Defaults to 20 values
        linearly spaced between 0.001 and 0.10.

    Returns
    -------
    pd.DataFrame with columns: contamination, precision, recall, f1.
    """
    if contamination_range is None:
        contamination_range = np.linspace(0.001, 0.10, 20)
    detector.fit(X_train)
    scores = detector.score(X_test)
    rows: list[dict[str, float]] = []
    for c in contamination_range:
        metrics = precision_recall_f1_at_contamination(y_test, scores, c)
        metrics["contamination"] = float(c)
        rows.append(metrics)
    return pd.DataFrame(rows)
