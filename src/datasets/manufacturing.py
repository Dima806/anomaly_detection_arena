"""Synthetic manufacturing sensor dataset with seasonal contextual anomalies."""

from __future__ import annotations

import numpy as np


def load_manufacturing(
    n_samples: int = 8000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic manufacturing temperature sensor dataset.

    Anomalies are **contextual / seasonal**: they are large positive temperature
    spikes that occur during overnight cooling windows (hours 0–6 of the daily
    cycle), when the baseline temperature is near its minimum.  The absolute
    temperature values are plausible but the residual from the seasonal
    expectation is very large.

    Parameters
    ----------
    n_samples:
        Total number of sensor readings.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5), dtype float64
        Features: raw_temp, hour_sin, hour_cos, seasonal_expected, residual.
    y : np.ndarray, shape (n_samples,), dtype int
        1 = anomaly, 0 = normal.
    """
    rng = np.random.default_rng(random_state)

    n_anomalies = int(n_samples * 0.015)
    n_normal = n_samples - n_anomalies

    # ------------------------------------------------------------------
    # Normal readings — dual-seasonal temperature model
    # ------------------------------------------------------------------
    t_normal = np.linspace(0, n_normal, n_normal)  # hours

    daily = 10 * np.sin(2 * np.pi * t_normal / 24)
    weekly = 5 * np.sin(2 * np.pi * t_normal / 168)
    seasonal_expected_normal = 40 + daily + weekly

    temp_normal = seasonal_expected_normal + rng.normal(0, 1.5, n_normal)
    residual_normal = temp_normal - seasonal_expected_normal  # ~N(0, 1.5)

    # ------------------------------------------------------------------
    # Anomalous readings — temperature spike during overnight cooling
    # Hours where daily cycle is most negative: t % 24 in [0, 6)
    # ------------------------------------------------------------------
    # Sample times uniformly from overnight windows across the full span
    candidate_t = np.arange(n_normal, dtype=float)
    overnight_mask = (candidate_t % 24) < 6
    overnight_t = candidate_t[overnight_mask]

    if len(overnight_t) < n_anomalies:
        # Fallback: repeat if somehow not enough candidates
        overnight_t = np.tile(overnight_t, n_anomalies // len(overnight_t) + 1)

    chosen_idx = rng.choice(len(overnight_t), size=n_anomalies, replace=False)
    t_anomaly = overnight_t[chosen_idx]

    daily_anom = 10 * np.sin(2 * np.pi * t_anomaly / 24)
    weekly_anom = 5 * np.sin(2 * np.pi * t_anomaly / 168)
    seasonal_expected_anomaly = 40 + daily_anom + weekly_anom

    temp_anomaly = seasonal_expected_anomaly + rng.normal(15, 2, n_anomalies)
    residual_anomaly = temp_anomaly - seasonal_expected_anomaly  # ~N(15, 2)

    # ------------------------------------------------------------------
    # Combine and build cyclical hour encoding
    # ------------------------------------------------------------------
    t_all = np.concatenate([t_normal, t_anomaly])
    raw_temp = np.concatenate([temp_normal, temp_anomaly])
    seasonal_expected = np.concatenate([seasonal_expected_normal, seasonal_expected_anomaly])
    residual = np.concatenate([residual_normal, residual_anomaly])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])

    hour = t_all % 24
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    X = np.column_stack([raw_temp, hour_sin, hour_cos, seasonal_expected, residual]).astype(
        np.float64
    )

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
