"""Focused tests for MahalanobisDetector.

Verifies the core mathematical properties:
- Contextual anomalies (multivariate outliers invisible in marginals) score higher
  than raw-univariate deviations that are not multivariate anomalies.
- The 5-line core distance function is numerically correct.
- Full fit + score + predict interface contract is satisfied.
"""

import numpy as np

from src.detectors.mahalanobis import MahalanobisDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_correlated_train(n: int = 200, seed: int = 0) -> np.ndarray:
    """Gaussian with strong positive correlation between features 0 and 1."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n, 5)
    # introduce correlation: feature 1 = feature 0 + small noise
    z[:, 1] = z[:, 0] + 0.05 * rng.randn(n)
    return z


# ---------------------------------------------------------------------------
# Core arithmetic correctness
# ---------------------------------------------------------------------------


class TestMahalanobisArithmetic:
    def test_identity_covariance_equals_euclidean(self) -> None:
        """When cov = I, Mahalanobis distance == Euclidean distance from mean."""
        rng = np.random.RandomState(1)
        # Build data that gives near-identity covariance
        X_train = rng.randn(500, 4)
        X_test = rng.randn(10, 4)
        det = MahalanobisDetector()
        det.fit(X_train)

        # Manual Euclidean distance from fitted mean
        mu = X_train.mean(axis=0)
        euclidean = np.linalg.norm(X_test - mu, axis=1)

        maha = det.score(X_test)
        # Should be approximately equal when cov ~ I
        np.testing.assert_allclose(maha, euclidean, rtol=0.2)

    def test_known_point_score(self) -> None:
        """Verify score for a single manually-constructed point."""
        # 2D data: mean=(0,0), identity covariance
        n = 1000
        rng = np.random.RandomState(2)
        X_train = rng.randn(n, 2)
        det = MahalanobisDetector()
        det.fit(X_train)

        # Point at (3, 0): expected Mahalanobis ~3
        point = np.array([[3.0, 0.0]])
        score = det.score(point)[0]
        assert abs(score - 3.0) < 0.5, f"Expected ~3.0, got {score:.3f}"

    def test_mean_point_scores_near_zero(self) -> None:
        """The sample mean should have Mahalanobis distance ~0."""
        rng = np.random.RandomState(3)
        X_train = rng.randn(100, 4)
        det = MahalanobisDetector()
        det.fit(X_train)
        mean_point = X_train.mean(axis=0, keepdims=True)
        score = det.score(mean_point)[0]
        assert score < 0.1, f"Mean should score near 0, got {score:.4f}"

    def test_scores_always_nonneg(self) -> None:
        rng = np.random.RandomState(4)
        X_train = rng.randn(80, 6)
        X_test = rng.randn(30, 6)
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert np.all(scores >= 0.0), f"Found negative scores: {scores[scores < 0]}"

    def test_further_point_scores_higher(self) -> None:
        rng = np.random.RandomState(5)
        X_train = rng.randn(100, 3)
        det = MahalanobisDetector()
        det.fit(X_train)
        near = np.array([[0.0, 0.0, 0.0]])
        far = np.array([[10.0, 10.0, 10.0]])
        assert det.score(near)[0] < det.score(far)[0]


# ---------------------------------------------------------------------------
# Contextual anomaly test
# ---------------------------------------------------------------------------


class TestContextualAnomalies:
    """Mahalanobis excels at detecting contextual (multivariate) anomalies.

    Consider data where feature A and feature B are tightly correlated.
    A point with (A=1, B=1) is normal; (A=1, B=-1) is anomalous even though
    both marginal values are unremarkable.  Euclidean / univariate detectors
    would miss this, but Mahalanobis should catch it.
    """

    def test_contextual_anomaly_scores_higher_than_marginal_outlier(self) -> None:
        """Contextual anomaly in correlated space must score higher than inlier."""
        X_train = _make_correlated_train(n=300, seed=10)
        det = MahalanobisDetector()
        det.fit(X_train)

        # Inlier: follows the correlation (feature 1 ≈ feature 0)
        X_inlier = np.array([[2.0, 2.0, 0.0, 0.0, 0.0]])

        # Contextual outlier: feature 0 and feature 1 are anti-correlated,
        # which is unusual given the training distribution.
        # Both marginal values (2.0 and -2.0) are within normal range, but
        # the combination violates the correlation structure.
        X_contextual = np.array([[2.0, -2.0, 0.0, 0.0, 0.0]])

        score_inlier = det.score(X_inlier)[0]
        score_contextual = det.score(X_contextual)[0]

        assert score_contextual > score_inlier, (
            f"Contextual anomaly ({score_contextual:.2f}) should score higher "
            f"than inlier ({score_inlier:.2f})"
        )

    def test_contextual_anomaly_flagged_by_predict(self) -> None:
        """predict() flags the contextual anomaly with high contamination rate."""
        X_train = _make_correlated_train(n=300, seed=11)
        det = MahalanobisDetector()
        det.fit(X_train)

        rng = np.random.RandomState(11)
        col0 = rng.randn(10)
        # Make proper inliers: col 1 = col 0 + tiny noise (follows correlation)
        X_inliers = np.column_stack(
            [
                col0,
                col0 + 0.05 * rng.randn(10),
                rng.randn(10),
                rng.randn(10),
                rng.randn(10),
            ]
        )

        col0b = rng.randn(10)
        # Make contextual outliers: col 1 deliberately anti-correlated with col 0
        X_outliers = np.column_stack(
            [
                col0b,
                -col0b + 0.05 * rng.randn(10),
                rng.randn(10),
                rng.randn(10),
                rng.randn(10),
            ]
        )

        X_combined = np.vstack([X_inliers, X_outliers])  # shape (20, 5)
        preds = det.predict(X_combined, contamination=0.5)

        outlier_flags = preds[10:]  # last 10 are contextual outliers
        inlier_flags = preds[:10]  # first 10 are inliers

        # The contextual outliers should be flagged more often than the inliers
        assert outlier_flags.sum() > inlier_flags.sum(), (
            f"Expected more outlier flags ({outlier_flags.sum()}) "
            f"than inlier flags ({inlier_flags.sum()})"
        )

    def test_engineered_residuals_score_higher(self) -> None:
        """Points that violate the correlation structure score higher than raw deviations.

        Engineered residual: a point whose raw feature values look normal
        (both near zero) but whose residual from the conditional mean is large.
        """
        X_train = _make_correlated_train(n=200, seed=99)
        det = MahalanobisDetector()
        det.fit(X_train)

        # Normal point at origin
        x_normal = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

        # Point with modest raw values but wrong correlation direction:
        # feature 0 = 1.5, feature 1 = -1.5 (raw values are not extreme, but
        # the pair (1.5, -1.5) is far from the regression line feature1 ~ feature0)
        x_residual_outlier = np.array([[1.5, -1.5, 0.0, 0.0, 0.0]])

        # Point with large raw deviation on feature 0 alone, consistent direction
        x_raw_deviation = np.array([[3.0, 3.0, 0.0, 0.0, 0.0]])

        score_normal = det.score(x_normal)[0]
        score_residual = det.score(x_residual_outlier)[0]
        score_raw = det.score(x_raw_deviation)[0]

        assert score_residual > score_normal, (
            f"Residual outlier ({score_residual:.2f}) should score > normal ({score_normal:.2f})"
        )
        # The anti-correlated pair (1.5, -1.5) violates structure more than
        # the in-structure deviation (3, 3), so should score higher.
        assert score_residual > score_raw, (
            f"Contextual residual anomaly ({score_residual:.2f}) should score "
            f"higher than raw in-structure deviation ({score_raw:.2f})"
        )


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


class TestMahalanobisInterface:
    def test_fit_score_predict_workflow(self) -> None:
        rng = np.random.RandomState(20)
        X_train = rng.randn(80, 5)
        X_test = rng.randn(20, 5)

        det = MahalanobisDetector()
        result = det.fit(X_train)
        assert result is det, "fit() must return self"

        scores = det.score(X_test)
        assert scores.shape == (20,)
        assert np.issubdtype(scores.dtype, np.floating)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

        preds = det.predict(X_test, contamination=0.1)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})
        assert np.issubdtype(preds.dtype, np.integer)

    def test_score_returns_float64(self) -> None:
        rng = np.random.RandomState(21)
        X_train = rng.randn(60, 4)
        X_test = rng.randn(15, 4)
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_name_attribute(self) -> None:
        assert MahalanobisDetector.name == "Mahalanobis"

    def test_initial_state_unfitted(self) -> None:
        det = MahalanobisDetector()
        assert det._mu is None
        assert det._cov_inv is None

    def test_fitted_state_populated(self) -> None:
        rng = np.random.RandomState(22)
        X_train = rng.randn(50, 3)
        det = MahalanobisDetector()
        det.fit(X_train)
        assert det._mu is not None
        assert det._cov_inv is not None
        assert det._mu.shape == (3,)
        assert det._cov_inv.shape == (3, 3)

    def test_predict_contamination_0_1(self) -> None:
        rng = np.random.RandomState(23)
        X_train = rng.randn(80, 5)
        X_test = rng.randn(20, 5)
        det = MahalanobisDetector()
        det.fit(X_train)
        for contamination in [0.05, 0.1, 0.2, 0.5]:
            preds = det.predict(X_test, contamination=contamination)
            assert set(np.unique(preds)).issubset({0, 1})

    def test_single_feature_data(self) -> None:
        rng = np.random.RandomState(24)
        X_train = rng.randn(60, 1)
        X_test = rng.randn(15, 1)
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert scores.shape == (15,)
        assert np.all(scores >= 0)

    def test_high_dimensional_data(self) -> None:
        rng = np.random.RandomState(25)
        X_train = rng.randn(200, 50)
        X_test = rng.randn(30, 50)
        det = MahalanobisDetector()
        det.fit(X_train)
        scores = det.score(X_test)
        assert scores.shape == (30,)
        assert not np.any(np.isnan(scores))
