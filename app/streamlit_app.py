"""Anomaly Detection Arena — interactive Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

matplotlib.use("Agg")

from src.datasets import load_manufacturing, load_network, make_synthetic_credit_card
from src.detectors import (
    AutoencoderDetector,
    DBSCANDetector,
    IsolationForestDetector,
    LOFDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
)
from src.evaluation.metrics import precision_recall_f1_at_contamination
from src.visualisation import plot_precision_recall_curve, plot_score_map

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Anomaly Detection Arena",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_OPTIONS = ["Network", "Manufacturing", "Credit Card (synthetic)"]
DETECTOR_OPTIONS = [
    "IsolationForest",
    "LOF",
    "OneClassSVM",
    "DBSCAN",
    "Mahalanobis",
    "Autoencoder",
]
CONTAMINATION_DEFAULTS = {
    "Network": 0.02,
    "Manufacturing": 0.015,
    "Credit Card (synthetic)": 0.0017,
}

# ---------------------------------------------------------------------------
# Cached helpers — contamination is NOT a cache key.
# Scores are fixed once the detector is fit; only the threshold changes.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading dataset…")
def get_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "Network":
        return load_network()
    if name == "Manufacturing":
        return load_manufacturing()
    return make_synthetic_credit_card(n_samples=2000, contamination=0.02)


@st.cache_data(show_spinner="Fitting detector and scoring…")
def get_scores(
    detector_name: str,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit detector on 80 % split, return (scores, y_test, X_test).

    Contamination is intentionally excluded from the cache key — scores are
    independent of the threshold.  The threshold is applied outside, so changing
    the contamination slider never re-fits the model.
    """
    X, y = get_dataset(dataset_name)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_test = y[split:]
    detector = _build_detector(detector_name)
    detector.fit(X_train)
    return detector.score(X_test), y_test, X_test


def _build_detector(name: str):
    if name == "IsolationForest":
        return IsolationForestDetector(n_estimators=100, random_state=42)
    if name == "LOF":
        return LOFDetector(n_neighbors=20)
    if name == "OneClassSVM":
        return OneClassSVMDetector(nu=0.05, max_samples=5000)
    if name == "DBSCAN":
        return DBSCANDetector(eps=0.5, min_samples=5)
    if name == "Mahalanobis":
        return MahalanobisDetector()
    return AutoencoderDetector(hidden_dim=16, epochs=20, random_state=42)


def _show_metrics(
    scores: np.ndarray,
    y_test: np.ndarray,
    contamination: float,
) -> None:
    """Render precision / recall / F1 / flagged-count metrics row."""
    m = precision_recall_f1_at_contamination(y_test, scores, contamination)
    n_flagged = int((scores >= np.quantile(scores, 1.0 - contamination)).sum())
    n_true = int(y_test.sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test samples", len(y_test))
    c2.metric("True anomalies", n_true)
    c3.metric("Flagged", n_flagged)
    c4.metric("Precision", f"{m['precision']:.3f}")
    c5.metric("Recall", f"{m['recall']:.3f}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Anomaly Detection Arena")
st.sidebar.markdown("Configure the experiment below.")

dataset_choice = st.sidebar.selectbox("Dataset", DATASET_OPTIONS, index=0)
detector_choice = st.sidebar.selectbox("Detector", DETECTOR_OPTIONS, index=0)

default_contamination = CONTAMINATION_DEFAULTS.get(dataset_choice, 0.02)
contamination = st.sidebar.slider(
    "Contamination rate",
    min_value=0.001,
    max_value=0.10,
    value=default_contamination,
    step=0.001,
    format="%.3f",
)
st.sidebar.caption(
    "Adjusting contamination re-draws all plots instantly — the model is not re-fit."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Comparison (Tab 3)**")
detector_a = st.sidebar.selectbox("Detector A", DETECTOR_OPTIONS, index=0, key="det_a")
detector_b = st.sidebar.selectbox("Detector B", DETECTOR_OPTIONS, index=1, key="det_b")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Anomaly Detection Arena")

tab1, tab2, tab3 = st.tabs(["Score Map", "PR Curve", "Detector Comparison"])

# ---------------------------------------------------------------------------
# Tab 1 — Score Map (threshold-based colours, updates with slider)
# ---------------------------------------------------------------------------

with tab1:
    st.subheader(f"{detector_choice} on {dataset_choice}")
    scores, y_test, X_test = get_scores(detector_choice, dataset_choice)

    # plot_score_map colours points by predicted label at current threshold
    fig = plot_score_map(
        X_test,
        scores,
        contamination=contamination,
        y_true=y_test,
        title=f"{detector_choice} — {dataset_choice}  (contamination={contamination:.3f})",
    )
    st.pyplot(fig)
    plt.close(fig)

    _show_metrics(scores, y_test, contamination)

# ---------------------------------------------------------------------------
# Tab 2 — PR Curve with operating-point marker
# ---------------------------------------------------------------------------

with tab2:
    st.subheader(f"PR Curve — {detector_choice} on {dataset_choice}")
    scores, y_test, X_test = get_scores(detector_choice, dataset_choice)

    if y_test.sum() == 0:
        st.warning("No positive labels in test split — cannot plot PR curve.")
    else:
        # Red dot moves along the curve as the slider changes
        fig = plot_precision_recall_curve(
            y_test,
            scores,
            label=detector_choice,
            contamination=contamination,
        )
        st.pyplot(fig)
        plt.close(fig)

    _show_metrics(scores, y_test, contamination)

# ---------------------------------------------------------------------------
# Tab 3 — Side-by-side comparison
# ---------------------------------------------------------------------------

with tab3:
    st.subheader(f"{detector_a} vs {detector_b} on {dataset_choice}")
    scores_a, y_test_a, X_test_a = get_scores(detector_a, dataset_choice)
    scores_b, y_test_b, X_test_b = get_scores(detector_b, dataset_choice)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**{detector_a}**")
        fig_a = plot_score_map(
            X_test_a,
            scores_a,
            contamination=contamination,
            y_true=y_test_a,
            title=detector_a,
        )
        st.pyplot(fig_a)
        plt.close(fig_a)

        if y_test_a.sum() > 0:
            fig_pr_a = plot_precision_recall_curve(
                y_test_a, scores_a, label=detector_a, contamination=contamination
            )
            st.pyplot(fig_pr_a)
            plt.close(fig_pr_a)

        _show_metrics(scores_a, y_test_a, contamination)

    with col_b:
        st.markdown(f"**{detector_b}**")
        fig_b = plot_score_map(
            X_test_b,
            scores_b,
            contamination=contamination,
            y_true=y_test_b,
            title=detector_b,
        )
        st.pyplot(fig_b)
        plt.close(fig_b)

        if y_test_b.sum() > 0:
            fig_pr_b = plot_precision_recall_curve(
                y_test_b, scores_b, label=detector_b, contamination=contamination
            )
            st.pyplot(fig_pr_b)
            plt.close(fig_pr_b)

        _show_metrics(scores_b, y_test_b, contamination)
