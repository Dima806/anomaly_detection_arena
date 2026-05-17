from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")


def plot_anomaly_scores(
    X: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    title: str = "Anomaly Scores",
) -> Figure:
    """Scatter plot of samples in 2-D PCA space, coloured by anomaly score.

    If y_true is provided, true anomalies are overlaid with blue X markers.

    Parameters
    ----------
    X:
        Feature matrix, shape (n, d).
    scores:
        Anomaly scores, shape (n,). Higher = more anomalous.
    y_true:
        Optional ground-truth labels (1=anomaly, 0=normal).
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure (caller is responsible for closing it).
    """
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
    else:
        # Edge case: pad to 2D
        coords = np.column_stack([X[:, 0], np.zeros(len(X))])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        cmap="YlOrRd",
        s=10,
        alpha=0.6,
        label="samples",
    )
    fig.colorbar(sc, ax=ax, label="Anomaly score")

    if y_true is not None:
        anomaly_idx = y_true == 1
        if anomaly_idx.any():
            ax.scatter(
                coords[anomaly_idx, 0],
                coords[anomaly_idx, 1],
                marker="x",
                c="blue",
                s=40,
                linewidths=1.5,
                label="true anomaly",
                zorder=5,
            )
            ax.legend(loc="upper right")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_score_map(
    X: np.ndarray,
    scores: np.ndarray,
    contamination: float,
    y_true: np.ndarray | None = None,
    title: str = "",
) -> Figure:
    """Scatter in PCA space coloured by threshold prediction, not raw score.

    Blue = predicted normal, red = flagged at current contamination threshold.
    Black X = true anomaly (if y_true provided). Updating contamination changes
    which points are red.
    """
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
    else:
        coords = np.column_stack([X[:, 0], np.zeros(len(X))])

    threshold = float(np.quantile(scores, 1.0 - contamination))
    flagged = scores >= threshold

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        coords[~flagged, 0],
        coords[~flagged, 1],
        c="steelblue",
        s=8,
        alpha=0.4,
        label=f"Normal ({(~flagged).sum()})",
    )
    if flagged.any():
        ax.scatter(
            coords[flagged, 0],
            coords[flagged, 1],
            c="tomato",
            s=22,
            alpha=0.85,
            label=f"Flagged ({flagged.sum()})",
        )
    if y_true is not None:
        true_anom = y_true == 1
        if true_anom.any():
            ax.scatter(
                coords[true_anom, 0],
                coords[true_anom, 1],
                marker="x",
                c="black",
                s=45,
                linewidths=1.5,
                zorder=5,
                label=f"True anomaly ({true_anom.sum()})",
            )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title or f"Predictions at contamination={contamination:.3f}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "",
    contamination: float | None = None,
) -> Figure:
    """Precision-recall curve with AUPRC displayed in the legend.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (1=anomaly).
    scores:
        Anomaly scores.
    label:
        Name prefix used in the legend entry.

    Returns
    -------
    matplotlib Figure.
    """
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    auprc = float(average_precision_score(y_true, scores))
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)

    legend_label = f"{label} (AUPRC={auprc:.3f})" if label else f"AUPRC={auprc:.3f}"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, lw=2, label=legend_label)

    if contamination is not None:
        thr = float(np.quantile(scores, 1.0 - contamination))
        y_pred = (scores >= thr).astype(int)
        op_p, op_r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        ax.scatter(
            [op_r],
            [op_p],
            s=120,
            c="red",
            zorder=6,
            label=f"@ contamination={contamination:.3f}  P={op_p:.2f} R={op_r:.2f}",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_arena_scoreboard(
    df: pd.DataFrame,
    metric: str = "auprc",
) -> Figure:
    """Heatmap of detector vs. dataset performance.

    Parameters
    ----------
    df:
        Scoreboard DataFrame as returned by ``run_arena``, with at minimum
        columns: name, dataset, and the requested metric column.
    metric:
        Column name of the metric to visualise.

    Returns
    -------
    matplotlib Figure.
    """
    pivot = df.pivot_table(index="name", columns="dataset", values=metric, aggfunc="mean")

    n_rows, n_cols = pivot.shape
    fig_width = max(6, n_cols * 1.5 + 2)
    fig_height = max(4, n_rows * 0.6 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.3 < val < 0.8 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    ax.set_title(f"Arena Scoreboard — {metric}")
    fig.tight_layout()
    return fig


def plot_contamination_sweep(
    df: pd.DataFrame,
    title: str = "Contamination Sweep",
) -> Figure:
    """Line plot showing precision, recall, and F1 over a contamination sweep.

    Parameters
    ----------
    df:
        DataFrame as returned by ``sweep_contamination``, with columns:
        contamination, precision, recall, f1.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["contamination"], df["precision"], marker="o", ms=4, label="Precision")
    ax.plot(df["contamination"], df["recall"], marker="s", ms=4, label="Recall")
    ax.plot(df["contamination"], df["f1"], marker="^", ms=4, label="F1")

    ax.set_xlabel("Contamination rate")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
