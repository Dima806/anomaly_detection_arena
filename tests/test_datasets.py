"""Tests for synthetic dataset loaders."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.credit_card import make_synthetic_credit_card
from src.datasets.manufacturing import load_manufacturing
from src.datasets.network import load_network

# ---------------------------------------------------------------------------
# make_synthetic_credit_card
# ---------------------------------------------------------------------------


class TestMakeSyntheticCreditCard:
    def test_shape(self) -> None:
        X, y = make_synthetic_credit_card(n_samples=200, contamination=0.05)
        assert X.shape == (200, 30), f"Expected (200, 30), got {X.shape}"
        assert y.shape == (200,), f"Expected (200,), got {y.shape}"

    def test_dtype(self) -> None:
        X, y = make_synthetic_credit_card(n_samples=200, contamination=0.05)
        assert X.dtype == np.float64
        assert np.issubdtype(y.dtype, np.integer)

    def test_labels_binary(self) -> None:
        _, y = make_synthetic_credit_card(n_samples=200, contamination=0.05)
        assert set(np.unique(y)).issubset({0, 1}), "Labels must be 0 or 1"

    def test_contamination_rate(self) -> None:
        n_samples = 2000
        contamination = 0.05
        _, y = make_synthetic_credit_card(
            n_samples=n_samples, contamination=contamination, random_state=0
        )
        actual_rate = y.mean()
        assert abs(actual_rate - contamination) < 0.02, (
            f"Contamination rate {actual_rate:.4f} too far from {contamination}"
        )

    def test_low_default_contamination(self) -> None:
        # Default contamination=0.0017, so with 1000 samples at least 1 fraud
        _, y = make_synthetic_credit_card(n_samples=1000, contamination=0.0017)
        assert y.sum() >= 1, "Should have at least one fraud with default settings"

    def test_reproducibility(self) -> None:
        X1, y1 = make_synthetic_credit_card(n_samples=100, random_state=7)
        X2, y2 = make_synthetic_credit_card(n_samples=100, random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self) -> None:
        X1, _ = make_synthetic_credit_card(n_samples=100, random_state=1)
        X2, _ = make_synthetic_credit_card(n_samples=100, random_state=2)
        assert not np.allclose(X1, X2), "Different seeds should produce different data"


# ---------------------------------------------------------------------------
# load_network
# ---------------------------------------------------------------------------


class TestLoadNetwork:
    def test_shape(self) -> None:
        X, y = load_network(n_samples=300)
        assert X.shape == (300, 5), f"Expected (300, 5), got {X.shape}"
        assert y.shape == (300,)

    def test_dtype(self) -> None:
        X, y = load_network(n_samples=300)
        assert X.dtype == np.float64
        assert np.issubdtype(y.dtype, np.integer)

    def test_labels_binary(self) -> None:
        _, y = load_network(n_samples=300)
        assert set(np.unique(y)).issubset({0, 1})

    def test_anomaly_rate(self) -> None:
        n_samples = 2000
        _, y = load_network(n_samples=n_samples, random_state=0)
        actual_rate = y.mean()
        assert abs(actual_rate - 0.02) < 0.01, f"Anomaly rate {actual_rate:.4f} too far from 0.02"

    def test_reproducibility(self) -> None:
        X1, y1 = load_network(n_samples=300, random_state=42)
        X2, y2 = load_network(n_samples=300, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_contextual_temporal_structure(self) -> None:
        """Anomalies should be concentrated in early-morning hours (1-5 AM).

        Feature layout: [hour_sin, hour_cos, request_rate, session_duration, subnet_score]
        We reconstruct hour = arctan2(hour_sin, hour_cos) * 24 / (2π) mod 24
        and verify anomalies are shifted toward overnight hours.
        """
        X, y = load_network(n_samples=3000, random_state=42)

        hour_sin = X[:, 0]
        hour_cos = X[:, 1]
        hour = (np.degrees(np.arctan2(hour_sin, hour_cos)) % 360) * 24 / 360

        anomaly_hours = hour[y == 1]
        normal_hours = hour[y == 0]

        # Anomalies should include overnight hours; use circular distance from 3 AM
        def mean_abs_deviation_from(arr: np.ndarray, target: float) -> float:
            diff = np.abs(arr - target)
            return float(np.minimum(diff, 24 - diff).mean())

        anom_dist = mean_abs_deviation_from(anomaly_hours, 3.0)
        normal_dist = mean_abs_deviation_from(normal_hours, 3.0)

        assert anom_dist < normal_dist, (
            f"Anomaly hours (mean dist from 3AM: {anom_dist:.2f}) should be "
            f"closer to 3AM than normal hours ({normal_dist:.2f})"
        )

    def test_anomalies_have_high_subnet_score(self) -> None:
        """Anomalies are designed with subnet_score ~ N(3, 0.5) vs N(0, ~0.5)."""
        X, y = load_network(n_samples=3000, random_state=42)
        subnet_score = X[:, 4]
        assert subnet_score[y == 1].mean() > subnet_score[y == 0].mean() + 1.0, (
            "Anomaly subnet scores should be substantially higher than normal"
        )


# ---------------------------------------------------------------------------
# load_manufacturing
# ---------------------------------------------------------------------------


class TestLoadManufacturing:
    def test_shape(self) -> None:
        X, y = load_manufacturing(n_samples=200)
        assert X.shape == (200, 5), f"Expected (200, 5), got {X.shape}"
        assert y.shape == (200,)

    def test_dtype(self) -> None:
        X, y = load_manufacturing(n_samples=200)
        assert X.dtype == np.float64
        assert np.issubdtype(y.dtype, np.integer)

    def test_labels_binary(self) -> None:
        _, y = load_manufacturing(n_samples=200)
        assert set(np.unique(y)).issubset({0, 1})

    def test_anomaly_rate(self) -> None:
        n_samples = 2000
        _, y = load_manufacturing(n_samples=n_samples, random_state=0)
        actual_rate = y.mean()
        assert abs(actual_rate - 0.015) < 0.01, (
            f"Anomaly rate {actual_rate:.4f} too far from 0.015"
        )

    def test_reproducibility(self) -> None:
        X1, y1 = load_manufacturing(n_samples=300, random_state=0)
        X2, y2 = load_manufacturing(n_samples=300, random_state=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_anomalies_have_high_residuals(self) -> None:
        """Feature layout: [raw_temp, hour_sin, hour_cos, seasonal_expected, residual].

        Anomaly residuals should be substantially larger than normal residuals.
        """
        X, y = load_manufacturing(n_samples=2000, random_state=42)
        residual = X[:, 4]

        normal_residual_mean = residual[y == 0].mean()
        anomaly_residual_mean = residual[y == 1].mean()

        assert anomaly_residual_mean > normal_residual_mean + 5.0, (
            f"Anomaly residual mean ({anomaly_residual_mean:.2f}) should be "
            f">> normal residual mean ({normal_residual_mean:.2f})"
        )

    def test_anomalies_occur_overnight(self) -> None:
        """Anomalies are placed in overnight cooling window (hour % 24 in [0, 6)).

        We reconstruct hour from hour_sin / hour_cos and check.
        """
        X, y = load_manufacturing(n_samples=3000, random_state=42)
        hour_sin = X[:, 1]
        hour_cos = X[:, 2]
        hour = (np.degrees(np.arctan2(hour_sin, hour_cos)) % 360) * 24 / 360

        anomaly_hours = hour[y == 1]
        # All anomaly hours should be within the 0–6 window
        overnight = (anomaly_hours < 6) | (anomaly_hours >= 18)
        # Allow a small tolerance in case of floating-point rounding at boundaries
        assert overnight.mean() > 0.9, (
            f"Expected >90% of anomalies overnight, got {overnight.mean():.2%}"
        )

    def test_feature_names_via_shape(self) -> None:
        """Sanity-check that exactly 5 features are returned."""
        X, _ = load_manufacturing(n_samples=200)
        assert X.shape[1] == 5


# ---------------------------------------------------------------------------
# Cross-dataset sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader,kwargs",
    [
        (make_synthetic_credit_card, {"n_samples": 250, "contamination": 0.05}),
        (load_network, {"n_samples": 250}),
        (load_manufacturing, {"n_samples": 250}),
    ],
)
def test_no_nans(loader, kwargs) -> None:  # type: ignore[no-untyped-def]
    X, y = loader(**kwargs)
    assert not np.any(np.isnan(X)), "X must not contain NaN values"
    assert not np.any(np.isnan(y.astype(float))), "y must not contain NaN values"


@pytest.mark.parametrize(
    "loader,kwargs",
    [
        (make_synthetic_credit_card, {"n_samples": 250, "contamination": 0.05}),
        (load_network, {"n_samples": 250}),
        (load_manufacturing, {"n_samples": 250}),
    ],
)
def test_no_infs(loader, kwargs) -> None:  # type: ignore[no-untyped-def]
    X, y = loader(**kwargs)
    assert not np.any(np.isinf(X)), "X must not contain Inf values"
