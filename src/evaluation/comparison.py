import numpy as np
import pandas as pd

from src.detectors.base import BaseDetector
from src.evaluation.metrics import evaluate_detector


def run_arena(
    detectors: list[BaseDetector],
    datasets: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> pd.DataFrame:
    """Run all detectors on all datasets.

    Parameters
    ----------
    detectors:
        List of detector instances to evaluate.
    datasets:
        Mapping of dataset name to (X, y, contamination) tuples.
        X and y are the full arrays; an 80/20 split is applied internally.

    Returns
    -------
    pd.DataFrame with columns: name, dataset, precision, recall, f1, auprc.
    """
    rows: list[dict[str, float | str]] = []
    for dataset_name, (X, y, contamination) in datasets.items():
        n = len(X)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        y_test = y[split:]
        for detector in detectors:
            result = evaluate_detector(detector, X_train, X_test, y_test, contamination)
            result["dataset"] = dataset_name
            rows.append(result)
    return pd.DataFrame(rows)
