"""Tests for src/evaluation/metrics.py and src/evaluation/contamination_sweep.py."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

from src.detectors.isolation_forest import IsolationForestDetector  # noqa: E402
from src.evaluation.contamination_sweep import sweep_contamination  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    auprc,
    evaluate_detector,
    precision_recall_f1_at_contamination,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n: int = 200,
    n_features: int = 5,
    contamination: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset: anomalies are offset by +5 in every feature."""
    rng = np.random.default_rng(random_state)
    n_anomalies = max(1, int(n * contamination))
    n_normal = n - n_anomalies
    X_normal = rng.standard_normal((n_normal, n_features))
    X_anomaly = rng.standard_normal((n_anomalies, n_features)) + 5.0
    X = np.vstack([X_normal, X_anomaly]).astype(np.float64)
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])
    return X, y


# ---------------------------------------------------------------------------
# precision_recall_f1_at_contamination
# ---------------------------------------------------------------------------


class TestPrecisionRecallF1AtContamination:
    def test_perfect_scores(self):
        # Scores perfectly separate anomalies (high score) from normals.
        y_true = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=2 / 6)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong_scores(self):
        # Scores are inverted: lowest scores are anomalies.
        y_true = np.array([1, 1, 0, 0, 0, 0])
        scores = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=2 / 6)
        # The top-contamination fraction (indices 2,3 or 3,4 etc.) are all normal
        assert result["recall"] == pytest.approx(0.0)

    def test_edge_contamination_near_zero(self):
        y_true = np.array([0, 0, 0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.99])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=0.001)
        # Only the very top score flagged
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_edge_contamination_near_one(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=0.99)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_return_keys(self):
        y_true = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=0.5)
        assert set(result.keys()) == {"precision", "recall", "f1"}

    def test_values_are_floats(self):
        y_true = np.array([0, 0, 1])
        scores = np.array([0.2, 0.3, 0.9])
        result = precision_recall_f1_at_contamination(y_true, scores, contamination=0.33)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# auprc
# ---------------------------------------------------------------------------


class TestAuprc:
    def test_perfect_classifier(self):
        # Perfect separation: anomalies always scored higher.
        y_true = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
        result = auprc(y_true, scores)
        assert result == pytest.approx(1.0)

    def test_random_classifier_near_contamination(self):
        # Random scores should yield AUPRC approximately equal to the
        # positive rate (i.e. contamination).
        rng = np.random.default_rng(42)
        n = 5000
        contamination = 0.05
        n_pos = int(n * contamination)
        y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n - n_pos, dtype=int)])
        scores = rng.uniform(0, 1, n)
        result = auprc(y_true, scores)
        # Allow generous tolerance for a random classifier
        assert result == pytest.approx(contamination, abs=0.03)

    def test_returns_float(self):
        y_true = np.array([0, 1])
        scores = np.array([0.2, 0.8])
        assert isinstance(auprc(y_true, scores), float)

    def test_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 50)
        scores = rng.uniform(0, 1, 50)
        result = auprc(y_true, scores)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate_detector
# ---------------------------------------------------------------------------


class TestEvaluateDetector:
    def test_returns_expected_keys(self):
        X, y = _make_data(n=200, n_features=5, contamination=0.1)
        split = int(0.8 * len(X))
        X_train, X_test, y_test = X[:split], X[split:], y[split:]
        detector = IsolationForestDetector(n_estimators=20, random_state=0)
        result = evaluate_detector(detector, X_train, X_test, y_test, contamination=0.1)
        assert set(result.keys()) == {"name", "precision", "recall", "f1", "auprc"}

    def test_name_field(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=10, random_state=1)
        result = evaluate_detector(detector, X[:split], X[split:], y[split:], 0.1)
        assert result["name"] == "IsolationForest"

    def test_metrics_in_unit_interval(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=2)
        result = evaluate_detector(detector, X[:split], X[split:], y[split:], 0.1)
        for key in ("precision", "recall", "f1", "auprc"):
            assert 0.0 <= float(result[key]) <= 1.0, f"{key} out of [0,1]"

    def test_high_auprc_on_separable_data(self):
        # Well-separated data should yield high AUPRC.
        X, y = _make_data(n=300, contamination=0.1, random_state=7)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=50, random_state=7)
        result = evaluate_detector(detector, X[:split], X[split:], y[split:], 0.1)
        assert float(result["auprc"]) > 0.5


# ---------------------------------------------------------------------------
# sweep_contamination
# ---------------------------------------------------------------------------


class TestSweepContamination:
    def test_default_row_count(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=3)
        df = sweep_contamination(detector, X[:split], X[split:], y[split:])
        # Default contamination_range has 20 points
        assert len(df) == 20

    def test_custom_range_row_count(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=4)
        custom_range = np.linspace(0.01, 0.09, 7)
        df = sweep_contamination(
            detector,
            X[:split],
            X[split:],
            y[split:],
            contamination_range=custom_range,
        )
        assert len(df) == 7

    def test_column_names(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=5)
        df = sweep_contamination(detector, X[:split], X[split:], y[split:])
        assert set(df.columns) == {"contamination", "precision", "recall", "f1"}

    def test_metrics_in_unit_interval(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=6)
        df = sweep_contamination(detector, X[:split], X[split:], y[split:])
        for col in ("precision", "recall", "f1"):
            assert df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"

    def test_contamination_column_matches_range(self):
        X, y = _make_data(n=200)
        split = int(0.8 * len(X))
        detector = IsolationForestDetector(n_estimators=20, random_state=8)
        custom_range = np.array([0.01, 0.05, 0.10])
        df = sweep_contamination(
            detector,
            X[:split],
            X[split:],
            y[split:],
            contamination_range=custom_range,
        )
        np.testing.assert_allclose(df["contamination"].values, custom_range)
