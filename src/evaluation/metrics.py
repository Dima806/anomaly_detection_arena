import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.detectors.base import BaseDetector


def precision_recall_f1_at_contamination(
    y_true: np.ndarray, scores: np.ndarray, contamination: float
) -> dict[str, float]:
    """Threshold scores at top-contamination quantile, return precision/recall/f1."""
    threshold = np.quantile(scores, 1.0 - contamination)
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall curve."""
    return float(average_precision_score(y_true, scores))


def evaluate_detector(
    detector: BaseDetector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float,
) -> dict[str, float | str]:
    """Fit detector, compute all metrics, return dict with name/precision/recall/f1/auprc."""
    detector.fit(X_train)
    scores = detector.score(X_test)
    pr = precision_recall_f1_at_contamination(y_test, scores, contamination)
    return {
        "name": detector.name,
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "auprc": auprc(y_test, scores),
    }
