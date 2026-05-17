"""Synthetic network intrusion dataset with contextual anomalies."""

from __future__ import annotations

import numpy as np


def load_network(
    n_samples: int = 10000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic network session dataset.

    Anomalies are **contextual**: they exhibit business-hours traffic patterns
    (high request rates) during the middle of the night (1–5 AM), which is
    suspicious even though each individual feature value is plausible.

    Parameters
    ----------
    n_samples:
        Total number of sessions.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5), dtype float64
        Features: hour_sin, hour_cos, request_rate, session_duration,
        subnet_score.
    y : np.ndarray, shape (n_samples,), dtype int
        1 = anomaly (intrusion), 0 = normal.
    """
    rng = np.random.default_rng(random_state)

    n_anomalies = int(n_samples * 0.02)
    n_normal = n_samples - n_anomalies

    # ------------------------------------------------------------------
    # Normal sessions
    # ------------------------------------------------------------------
    hours_normal = rng.integers(0, 24, size=n_normal)
    business = (hours_normal >= 9) & (hours_normal < 17)

    request_rate_normal = np.where(
        business,
        np.clip(rng.normal(100, 15, n_normal), 1, None),
        np.clip(rng.normal(10, 3, n_normal), 1, None),
    )

    session_duration_normal = np.clip(
        request_rate_normal / rng.normal(5, 0.5, n_normal) + rng.normal(0, 1, n_normal),
        0.1,
        None,
    )

    subnet_score_normal = np.where(
        business,
        rng.normal(0, 0.5, n_normal),
        rng.normal(0, 0.1, n_normal),
    )

    # ------------------------------------------------------------------
    # Anomalous sessions — contextual: high traffic at 1–5 AM
    # ------------------------------------------------------------------
    hours_anomaly = rng.choice([1, 2, 3, 4, 5], size=n_anomalies)

    request_rate_anomaly = np.clip(rng.normal(100, 15, n_anomalies), 1, None)

    session_duration_anomaly = np.clip(
        rng.normal(1, 0.5, n_anomalies),
        0.1,
        None,
    )

    subnet_score_anomaly = rng.normal(3, 0.5, n_anomalies)

    # ------------------------------------------------------------------
    # Combine and compute cyclical hour encoding
    # ------------------------------------------------------------------
    hours = np.concatenate([hours_normal, hours_anomaly]).astype(np.float64)
    request_rate = np.concatenate([request_rate_normal, request_rate_anomaly])
    session_duration = np.concatenate([session_duration_normal, session_duration_anomaly])
    subnet_score = np.concatenate([subnet_score_normal, subnet_score_anomaly])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])

    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    X = np.column_stack([hour_sin, hour_cos, request_rate, session_duration, subnet_score]).astype(
        np.float64
    )

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
