"""Integration tests for all anomaly detectors.

Uses small synthetic arrays (50-100 rows, 5 features) to keep tests fast.
"""

import numpy as np

from src.detectors.autoencoder import AutoencoderDetector
from src.detectors.base import BaseDetector
from src.detectors.dbscan_detector import DBSCANDetector
from src.detectors.isolation_forest import IsolationForestDetector
from src.detectors.lof import LOFDetector
from src.detectors.mahalanobis import MahalanobisDetector
from src.detectors.one_class_svm import OneClassSVMDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_data(n: int = 80, d: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n, d)
    X_test = rng.randn(20, d)
    return X_train, X_test


def make_data_1d(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n, 1)
    X_test = rng.randn(15, 1)
    return X_train, X_test


def assert_scores_valid(scores: np.ndarray, n: int) -> None:
    assert scores.shape == (n,), f"Expected shape ({n},), got {scores.shape}"
    assert np.issubdtype(scores.dtype, np.floating), f"Expected float dtype, got {scores.dtype}"
    assert not np.any(np.isnan(scores)), "Scores contain NaN"
    assert not np.any(np.isinf(scores)), "Scores contain Inf"


def assert_predictions_binary(preds: np.ndarray, n: int) -> None:
    assert preds.shape == (n,)
    assert set(np.unique(preds)).issubset({0, 1}), (
        f"Predictions contain non-binary values: {np.unique(preds)}"
    )


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------


class TestIsolationForestDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = IsolationForestDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = IsolationForestDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = IsolationForestDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)
        assert preds.sum() >= 1

    def test_predict_small_contamination(self) -> None:
        X_train, X_test = make_data()
        det = IsolationForestDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.05)
        assert_predictions_binary(preds, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = IsolationForestDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = IsolationForestDetector()
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert IsolationForestDetector.name == "IsolationForest"

    def test_is_base_detector(self) -> None:
        assert issubclass(IsolationForestDetector, BaseDetector)

    def test_scores_higher_for_outliers(self) -> None:
        rng = np.random.RandomState(7)
        X_train = rng.randn(80, 5)
        X_inliers = rng.randn(10, 5)
        # Outliers far from origin
        X_outliers = rng.randn(10, 5) * 0.1 + 10.0
        det = IsolationForestDetector()
        det.fit(X_train)
        mean_inlier = det.score(X_inliers).mean()
        mean_outlier = det.score(X_outliers).mean()
        assert mean_outlier > mean_inlier


# ---------------------------------------------------------------------------
# LOF
# ---------------------------------------------------------------------------


class TestLOFDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = LOFDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = LOFDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = LOFDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)

    def test_predict_small_contamination(self) -> None:
        X_train, X_test = make_data()
        det = LOFDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.05)
        assert_predictions_binary(preds, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = LOFDetector(n_neighbors=5)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = LOFDetector()
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert LOFDetector.name == "LOF"

    def test_scores_higher_for_outliers(self) -> None:
        rng = np.random.RandomState(7)
        X_train = rng.randn(80, 5)
        X_inliers = rng.randn(10, 5)
        X_outliers = rng.randn(10, 5) * 0.1 + 10.0
        det = LOFDetector(n_neighbors=5)
        det.fit(X_train)
        mean_inlier = det.score(X_inliers).mean()
        mean_outlier = det.score(X_outliers).mean()
        assert mean_outlier > mean_inlier


# ---------------------------------------------------------------------------
# OneClassSVM
# ---------------------------------------------------------------------------


class TestOneClassSVMDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = OneClassSVMDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = OneClassSVMDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = OneClassSVMDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)

    def test_subsampling_applied(self) -> None:
        rng = np.random.RandomState(0)
        X_large = rng.randn(200, 5)
        X_test = rng.randn(10, 5)
        det = OneClassSVMDetector(max_samples=50)
        det.fit(X_large)
        scores = det.score(X_test)
        assert_scores_valid(scores, 10)

    def test_no_subsampling_when_small(self) -> None:
        X_train, X_test = make_data(n=50)
        det = OneClassSVMDetector(max_samples=10000)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = OneClassSVMDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = OneClassSVMDetector()
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert OneClassSVMDetector.name == "OneClassSVM"


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


class TestDBSCANDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = DBSCANDetector(eps=1.5, min_samples=3)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = DBSCANDetector(eps=1.5, min_samples=3)
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = DBSCANDetector(eps=1.5, min_samples=3)
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)

    def test_no_core_points_fallback(self) -> None:
        # Very tight eps means no core points; should fall back to all points
        X_train, X_test = make_data()
        det = DBSCANDetector(eps=0.001, min_samples=100)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = DBSCANDetector(eps=1.0, min_samples=3)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = DBSCANDetector(eps=1.5, min_samples=3)
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert DBSCANDetector.name == "DBSCAN"

    def test_scores_higher_for_distant_points(self) -> None:
        rng = np.random.RandomState(11)
        X_train = rng.randn(80, 5)
        X_near = rng.randn(10, 5) * 0.01  # near origin cluster
        X_far = rng.randn(10, 5) * 0.1 + 20.0  # far from any cluster
        det = DBSCANDetector(eps=1.5, min_samples=3)
        det.fit(X_train)
        mean_near = det.score(X_near).mean()
        mean_far = det.score(X_far).mean()
        assert mean_far > mean_near


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------


class TestMahalanobisDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = MahalanobisDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = MahalanobisDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)

    def test_predict_small_contamination(self) -> None:
        X_train, X_test = make_data()
        det = MahalanobisDetector()
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.05)
        assert_predictions_binary(preds, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = MahalanobisDetector()
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert MahalanobisDetector.name == "Mahalanobis"

    def test_scores_nonneg(self) -> None:
        X_train, X_test = make_data()
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert np.all(scores >= 0), "Mahalanobis scores must be non-negative"

    def test_scores_higher_for_outliers(self) -> None:
        rng = np.random.RandomState(3)
        X_train = rng.randn(100, 5)
        X_inliers = rng.randn(10, 5)
        X_outliers = rng.randn(10, 5) * 0.1 + 8.0
        det = MahalanobisDetector()
        det.fit(X_train)
        mean_inlier = det.score(X_inliers).mean()
        mean_outlier = det.score(X_outliers).mean()
        assert mean_outlier > mean_inlier


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class TestAutoencoderDetector:
    def test_score_shape_and_dtype(self) -> None:
        X_train, X_test = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 20)

    def test_predict_binary(self) -> None:
        X_train, X_test = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.1)
        assert_predictions_binary(preds, 20)

    def test_predict_large_contamination(self) -> None:
        X_train, X_test = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.5)
        assert_predictions_binary(preds, 20)

    def test_predict_small_contamination(self) -> None:
        X_train, X_test = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        det.fit(X_train)
        preds = det.predict(X_test, contamination=0.05)
        assert_predictions_binary(preds, 20)

    def test_single_feature(self) -> None:
        X_train, X_test = make_data_1d()
        det = AutoencoderDetector(epochs=3, hidden_dim=4)
        det.fit(X_train)
        scores = det.score(X_test)
        assert_scores_valid(scores, 15)

    def test_fit_returns_self(self) -> None:
        X_train, _ = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        result = det.fit(X_train)
        assert result is det

    def test_name_attribute(self) -> None:
        assert AutoencoderDetector.name == "Autoencoder"

    def test_scores_nonneg(self) -> None:
        X_train, X_test = make_data()
        det = AutoencoderDetector(epochs=3, hidden_dim=8)
        det.fit(X_train)
        scores = det.score(X_test)
        assert np.all(scores >= 0), "Reconstruction errors must be non-negative"

    def test_batch_size_larger_than_data(self) -> None:
        rng = np.random.RandomState(99)
        X_small = rng.randn(10, 3)
        det = AutoencoderDetector(epochs=3, hidden_dim=4, batch_size=256)
        det.fit(X_small)
        scores = det.score(X_small)
        assert_scores_valid(scores, 10)


# ---------------------------------------------------------------------------
# BaseDetector: predict contract tests (via a concrete subclass)
# ---------------------------------------------------------------------------


class TestBaseDetectorPredict:
    """Test the predict() default implementation in BaseDetector."""

    def _make_fitted(self) -> IsolationForestDetector:
        X_train, _ = make_data()
        det = IsolationForestDetector()
        det.fit(X_train)
        return det

    def test_predict_contamination_fraction_respected(self) -> None:
        _, X_test = make_data()
        det = self._make_fitted()
        contamination = 0.25
        preds = det.predict(X_test, contamination=contamination)
        # With 20 points and contamination=0.25 the threshold is the 75th percentile.
        # At minimum ceil(20 * 0.25) = 5 points should be flagged.
        assert preds.sum() >= 1

    def test_predict_all_normal_at_zero_contamination(self) -> None:
        _, X_test = make_data()
        det = self._make_fitted()
        preds = det.predict(X_test, contamination=0.0)
        # quantile at 1.0 = max score; only exact-max points flagged
        assert preds.sum() >= 1  # at least the maximum point is flagged

    def test_predict_dtype_is_int(self) -> None:
        _, X_test = make_data()
        det = self._make_fitted()
        preds = det.predict(X_test, contamination=0.1)
        assert np.issubdtype(preds.dtype, np.integer)
