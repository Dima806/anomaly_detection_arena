"""Credit card fraud dataset loader and synthetic generator."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "creditcard.csv"
_DOWNLOAD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def make_synthetic_credit_card(
    n_samples: int = 1000,
    contamination: float = 0.0017,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking credit card fraud patterns.

    Produces 30 float features (V1-V28, Amount, Time) with a low fraud rate.
    No network access — suitable for unit tests.

    Parameters
    ----------
    n_samples:
        Total number of transactions to generate.
    contamination:
        Fraction of fraudulent transactions (label=1).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 30), dtype float64
    y : np.ndarray, shape (n_samples,), dtype int, values in {0, 1}
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_samples * contamination))
    n_normal = n_samples - n_fraud

    # Normal transactions: PCA-like components centred near zero
    X_normal = rng.standard_normal((n_normal, 28))
    amount_normal = rng.exponential(scale=88.0, size=(n_normal, 1))
    time_normal = np.linspace(0, 172792, n_normal).reshape(-1, 1) + rng.normal(
        0, 60, (n_normal, 1)
    )
    X_normal = np.hstack([X_normal, amount_normal, time_normal])

    # Fraud transactions: shifted distribution (higher amounts, different PCA)
    X_fraud = rng.standard_normal((n_fraud, 28)) * 1.5 + 0.5
    amount_fraud = rng.exponential(scale=122.0, size=(n_fraud, 1))
    time_fraud = rng.uniform(0, 172792, (n_fraud, 1))
    X_fraud = np.hstack([X_fraud, amount_fraud, time_fraud])

    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    # Normalise Amount (col 28) and Time (col 29)
    scaler = StandardScaler()
    X[:, 28:30] = scaler.fit_transform(X[:, 28:30])

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx].astype(np.float64), y[idx]


def load_credit_card(
    data_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the Kaggle credit card fraud CSV.

    Parameters
    ----------
    data_path:
        Path to the CSV file. Defaults to ``data/creditcard.csv`` relative to
        the project root.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 30), dtype float64
        Features V1-V28, Amount, Time (Amount and Time are StandardScaler-
        normalised).
    y : np.ndarray, shape (n_samples,), dtype int
        1 = fraud, 0 = normal.
    """
    csv_path = data_path or _DEFAULT_DATA_PATH

    df = pd.read_csv(csv_path)

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["Class"].values.astype(int)

    scaler = StandardScaler()
    amount_time_idx = [feature_cols.index("Amount"), feature_cols.index("Time")]
    X[:, amount_time_idx] = scaler.fit_transform(X[:, amount_time_idx])

    return X, y


if __name__ == "__main__":
    dest = _DEFAULT_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading credit card dataset to {dest} …")
    urllib.request.urlretrieve(_DOWNLOAD_URL, dest)
    print("Download complete.")
    X, y = load_credit_card(dest)
    print(f"Loaded: X={X.shape}, fraud rate={y.mean():.4f}")
