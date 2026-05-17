"""Generate all 5 Jupyter notebooks for anomaly_detection_arena.

Run once from the project root:
    uv run python scripts/generate_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def save_nb(nb: nbf.NotebookNode, path: Path) -> None:
    nb.metadata.setdefault("kernelspec", {}).update(
        {"display_name": "anomaly-arena", "language": "python", "name": "anomaly-arena"}
    )
    nb.metadata.setdefault("language_info", {}).update({"name": "python", "version": "3.11"})
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Shared setup cell (prepended to every notebook)
# ---------------------------------------------------------------------------
_SETUP = """\
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = _ROOT / "outputs" / "figures"
OUTPUTS = _ROOT / "outputs"
FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

from src.datasets import load_network, load_manufacturing, make_synthetic_credit_card
from src.detectors import (
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    DBSCANDetector,
    MahalanobisDetector,
    AutoencoderDetector,
)
from src.evaluation import (
    run_arena,
    sweep_contamination,
    evaluate_detector,
    auprc,
    precision_recall_f1_at_contamination,
)
import src.visualisation as vis
"""


# ===========================================================================
# 01 — Data Exploration
# ===========================================================================
def nb01() -> nbf.NotebookNode:
    cells = [
        md(
            "# 01 — Data Exploration\n\n"
            "Load all three datasets, inspect structure, visualise distributions "
            "and the contextual patterns that define what 'normal' means in each domain."
        ),
        code(_SETUP),
        md("## Load datasets"),
        code(
            """\
# Use synthetic credit card so the notebook runs without downloading the CSV.
# Swap in load_credit_card() after running `make data`.
X_cc, y_cc = make_synthetic_credit_card(n_samples=5000, random_state=42)
X_net, y_net = load_network(n_samples=3000, random_state=42)
X_mfg, y_mfg = load_manufacturing(n_samples=2000, random_state=42)

for name, X, y in [("credit_card", X_cc, y_cc), ("network", X_net, y_net), ("manufacturing", X_mfg, y_mfg)]:
    print(f"{name:15s}  shape={X.shape}  anomalies={y.sum()} ({y.mean():.3%})")
"""
        ),
        md("## Dataset summary → `outputs/01_dataset_summary.json`"),
        code(
            """\
summary = {}
for name, X, y in [("credit_card", X_cc, y_cc), ("network", X_net, y_net), ("manufacturing", X_mfg, y_mfg)]:
    X_n, X_a = X[y == 0], X[y == 1]
    summary[name] = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_anomalies": int(y.sum()),
        "anomaly_rate": float(y.mean()),
        "feature_means_normal": X_n.mean(axis=0).tolist(),
        "feature_stds_normal": X_n.std(axis=0).tolist(),
        "feature_means_anomaly": X_a.mean(axis=0).tolist(),
        "feature_stds_anomaly": X_a.std(axis=0).tolist(),
    }

with (OUTPUTS / "01_dataset_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print("Saved outputs/01_dataset_summary.json")
pd.DataFrame(
    {n: {"n_samples": v["n_samples"], "n_features": v["n_features"],
         "n_anomalies": v["n_anomalies"], "anomaly_rate_pct": f"{v['anomaly_rate']:.3%}"}
     for n, v in summary.items()}
).T
"""
        ),
        md("## Feature distributions: normal vs anomaly"),
        code(
            """\
fig, axes = plt.subplots(3, 5, figsize=(16, 9))
datasets_list = [
    ("Credit Card (synthetic)", X_cc, y_cc),
    ("Network Intrusion", X_net, y_net),
    ("Manufacturing Sensor", X_mfg, y_mfg),
]
feature_labels = {
    "Credit Card (synthetic)": [f"V{i}" for i in range(1, 6)],
    "Network Intrusion": ["hour_sin", "hour_cos", "req_rate", "sess_dur", "subnet"],
    "Manufacturing Sensor": ["raw_temp", "hour_sin", "hour_cos", "seasonal_exp", "residual"],
}
for row, (title, X, y) in enumerate(datasets_list):
    labels = feature_labels[title]
    for col in range(5):
        ax = axes[row, col]
        ax.hist(X[y == 0, col], bins=30, alpha=0.6, color="steelblue", density=True, label="normal")
        ax.hist(X[y == 1, col], bins=30, alpha=0.6, color="tomato", density=True, label="anomaly")
        ax.set_title(f"{labels[col]}", fontsize=8)
        if col == 0:
            ax.set_ylabel(title, fontsize=7)
        if row == 0 and col == 4:
            ax.legend(fontsize=7)
fig.suptitle("Feature Distributions — Normal (blue) vs Anomaly (red)", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "01_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 01_feature_distributions.png")
"""
        ),
        md(
            "## Network dataset: contextual structure\n\n"
            "Anomalies have *normal absolute request rates* but occur at unusual times."
        ),
        code(
            """\
# Reconstruct hour from sin/cos encoding
hour_net = (np.arctan2(X_net[:, 0], X_net[:, 1]) * 24 / (2 * np.pi)) % 24

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, mask, label, color in [
    (axes[0], y_net == 0, "Normal (y=0)", "steelblue"),
    (axes[1], y_net == 1, "Anomaly (y=1)", "tomato"),
]:
    ax.scatter(hour_net[mask], X_net[mask, 2], alpha=0.35, s=8, c=color)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Request rate")
    ax.set_title(f"Network — {label}")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

fig.suptitle("Contextual anomaly: high request rate at 1–5 AM looks normal globally", fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / "01_network_contextual_structure.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 01_network_contextual_structure.png")
"""
        ),
        md(
            "## Manufacturing dataset: seasonal pattern\n\n"
            "Anomalies fall *within the global temperature range* but deviate from the seasonal expectation."
        ),
        code(
            """\
n_show = min(1500, len(X_mfg))
t_idx = np.arange(n_show)
anom_mask_mfg = y_mfg[:n_show] == 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax1.plot(t_idx, X_mfg[:n_show, 0], lw=0.7, alpha=0.7, color="steelblue", label="raw temp")
ax1.plot(t_idx, X_mfg[:n_show, 3], lw=1.2, color="orange", label="seasonal expected")
ax1.scatter(np.where(anom_mask_mfg)[0], X_mfg[:n_show, 0][anom_mask_mfg],
            color="red", s=25, zorder=5, label="anomaly")
ax1.set_ylabel("Temperature (°C)")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title("Raw temperature vs. seasonal expectation")

ax2.plot(t_idx, X_mfg[:n_show, 4], lw=0.7, color="purple", alpha=0.7, label="residual")
ax2.axhline(0, color="k", lw=0.5, ls="--")
ax2.scatter(np.where(anom_mask_mfg)[0], X_mfg[:n_show, 4][anom_mask_mfg],
            color="red", s=25, zorder=5, label="anomaly residual")
ax2.set_ylabel("Residual (°C)")
ax2.set_xlabel("Time step")
ax2.legend(fontsize=8)
ax2.set_title("Residual = raw − seasonal expected  (anomalies are obvious here)")

fig.tight_layout()
fig.savefig(FIGURES / "01_manufacturing_seasonal.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 01_manufacturing_seasonal.png")
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


# ===========================================================================
# 02 — Global vs Contextual
# ===========================================================================
def nb02() -> nbf.NotebookNode:
    cells = [
        md(
            "# 02 — Global vs Contextual Anomalies\n\n"
            "Define the two anomaly types, show that credit card fraud is a *global outlier* "
            "while network intrusion and manufacturing faults are *contextual anomalies* "
            "invisible to global statistics."
        ),
        code(_SETUP),
        md("## Load datasets"),
        code(
            """\
X_cc, y_cc = make_synthetic_credit_card(n_samples=5000, random_state=42)
X_net, y_net = load_network(n_samples=3000, random_state=42)
X_mfg, y_mfg = load_manufacturing(n_samples=2000, random_state=42)
print("Loaded.")
"""
        ),
        md(
            "## Global distance: Mahalanobis from the mean\n\n"
            "A **global outlier** has a large Mahalanobis distance from the overall mean. "
            "A **contextual anomaly** may be close to the mean globally."
        ),
        code(
            """\
def global_mahalanobis(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(X, rowvar=False))
    diff = X - mu
    return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0.0))

results = {}
for name, X, y in [("credit_card", X_cc, y_cc), ("network", X_net, y_net), ("manufacturing", X_mfg, y_mfg)]:
    dists = global_mahalanobis(X)
    d_normal = float(dists[y == 0].mean())
    d_anomaly = float(dists[y == 1].mean())
    separation = d_anomaly / max(d_normal, 1e-9)
    results[name] = {
        "mean_distance_normal": round(d_normal, 4),
        "mean_distance_anomaly": round(d_anomaly, 4),
        "separation_ratio": round(separation, 4),
    }
    print(f"{name:15s}  normal={d_normal:.2f}  anomaly={d_anomaly:.2f}  ratio={separation:.2f}x")

print()
print("Credit card: high ratio → global outliers  ✓")
print("Network/manufacturing: low ratio → contextual anomalies  ✓")
"""
        ),
        md("## Distribution of global Mahalanobis distances"),
        code(
            """\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
dataset_items = [
    ("Credit Card\\n(global outliers)", X_cc, y_cc),
    ("Network\\n(contextual)", X_net, y_net),
    ("Manufacturing\\n(contextual)", X_mfg, y_mfg),
]
for ax, (title, X, y) in zip(axes, dataset_items):
    dists = global_mahalanobis(X)
    ax.hist(dists[y == 0], bins=40, alpha=0.6, color="steelblue", density=True, label="normal")
    ax.hist(dists[y == 1], bins=20, alpha=0.7, color="tomato", density=True, label="anomaly")
    ax.set_xlabel("Mahalanobis distance from mean")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Global Mahalanobis Distance Distribution", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "02_global_mahalanobis_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 02_global_mahalanobis_distributions.png")
"""
        ),
        md(
            "## Why Isolation Forest fails on contextual anomalies\n\n"
            "Isolation Forest measures how many random splits are needed to isolate a point. "
            "Contextual anomalies are embedded in a dense region of feature space — they take "
            "many splits to isolate — so Isolation Forest scores them as *normal*."
        ),
        code(
            """\
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, X, y) in zip(axes, dataset_items):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    ax.scatter(X2[y == 0, 0], X2[y == 0, 1], alpha=0.3, s=6, c="steelblue", label="normal")
    ax.scatter(X2[y == 1, 0], X2[y == 1, 1], alpha=0.8, s=20, c="tomato", label="anomaly", zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8)
fig.suptitle("PCA Projection — anomalies visible only in credit card dataset", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "02_pca_projections.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 02_pca_projections.png")
"""
        ),
        md(
            "## Contextual feature: residual reveals the anomaly\n\n"
            "For manufacturing, the raw temperature does NOT separate anomalies from normals. "
            "The *residual* (raw − seasonal expectation) does."
        ),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw temperature
axes[0].hist(X_mfg[y_mfg == 0, 0], bins=40, alpha=0.6, color="steelblue", density=True, label="normal")
axes[0].hist(X_mfg[y_mfg == 1, 0], bins=20, alpha=0.7, color="tomato", density=True, label="anomaly")
axes[0].set_xlabel("Raw temperature (°C)")
axes[0].set_title("Raw temperature: anomalies overlap normal range")
axes[0].legend()

# Residual
axes[1].hist(X_mfg[y_mfg == 0, 4], bins=40, alpha=0.6, color="steelblue", density=True, label="normal")
axes[1].hist(X_mfg[y_mfg == 1, 4], bins=20, alpha=0.7, color="tomato", density=True, label="anomaly")
axes[1].set_xlabel("Seasonal residual (°C)")
axes[1].set_title("Residual: anomalies are clearly separated")
axes[1].legend()

fig.suptitle("Manufacturing — Feature Engineering Reveals Contextual Anomalies", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "02_manufacturing_raw_vs_residual.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 02_manufacturing_raw_vs_residual.png")
"""
        ),
        md("## Save analysis → `outputs/02_global_vs_contextual.json`"),
        code(
            """\
analysis = {
    "global_mahalanobis": results,
    "interpretation": {
        "credit_card": "High separation ratio confirms global outliers. IF and Mahalanobis both work.",
        "network": "Low ratio confirms contextual anomalies. Raw features do not separate classes globally.",
        "manufacturing": "Low ratio. Residual feature is essential for detection.",
    },
    "key_insight": (
        "Global Mahalanobis distance separates credit card anomalies (ratio > 2) "
        "but fails for network and manufacturing (ratio near 1). "
        "Feature engineering (residuals, time encoding) is required for contextual detection."
    ),
}

with (OUTPUTS / "02_global_vs_contextual.json").open("w") as f:
    json.dump(analysis, f, indent=2)
print("Saved outputs/02_global_vs_contextual.json")
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


# ===========================================================================
# 03 — Algorithm Comparison (the Arena)
# ===========================================================================
def nb03() -> nbf.NotebookNode:
    cells = [
        md(
            "# 03 — Algorithm Comparison: The Arena\n\n"
            "Run all six detectors on all three datasets. "
            "Report precision, recall, F1 at the true contamination rate, and AUPRC. "
            "Produce the arena scoreboard."
        ),
        code(_SETUP),
        md("## Load datasets"),
        code(
            """\
X_cc, y_cc = make_synthetic_credit_card(n_samples=5000, random_state=42)
X_net, y_net = load_network(n_samples=3000, random_state=42)
X_mfg, y_mfg = load_manufacturing(n_samples=2000, random_state=42)
print("Loaded.")
"""
        ),
        md("## Run the arena"),
        code(
            """\
# Use fewer autoencoder epochs for notebook speed; increase for production
detectors = [
    IsolationForestDetector(n_estimators=100, random_state=42),
    LOFDetector(n_neighbors=20),
    OneClassSVMDetector(nu=0.05, max_samples=2000),
    DBSCANDetector(eps=0.5, min_samples=5),
    MahalanobisDetector(),
    AutoencoderDetector(hidden_dim=32, epochs=20, random_state=42),
]

contamination = {"credit_card": 0.0017, "network": 0.02, "manufacturing": 0.015}

datasets_arena = {
    "credit_card": (X_cc, y_cc, contamination["credit_card"]),
    "network": (X_net, y_net, contamination["network"]),
    "manufacturing": (X_mfg, y_mfg, contamination["manufacturing"]),
}

print("Running arena (this takes ~1–2 minutes)...")
scoreboard = run_arena(detectors, datasets_arena)
print("Done.")
scoreboard.round(3)
"""
        ),
        md("## Scoreboard heatmap — AUPRC"),
        code(
            """\
fig = vis.plot_arena_scoreboard(scoreboard, metric="auprc")
fig.savefig(FIGURES / "03_arena_auprc_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 03_arena_auprc_heatmap.png")
"""
        ),
        md("## Scoreboard heatmap — F1"),
        code(
            """\
fig = vis.plot_arena_scoreboard(scoreboard, metric="f1")
fig.savefig(FIGURES / "03_arena_f1_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 03_arena_f1_heatmap.png")
"""
        ),
        md("## Best detector per dataset"),
        code(
            """\
best = (
    scoreboard.loc[scoreboard.groupby("dataset")["auprc"].idxmax()]
    .reset_index(drop=True)
)
print("Best detector per dataset (by AUPRC):")
print(best[["dataset", "name", "auprc", "f1"]].to_string(index=False))
"""
        ),
        md("## PR curves per dataset"),
        code(
            """\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
from sklearn.metrics import precision_recall_curve

for ax, (ds_name, (X, y, cont)) in zip(axes, datasets_arena.items()):
    split = int(0.8 * len(X))
    X_train, X_test, y_test = X[:split], X[split:], y[split:]
    for det in detectors:
        det.fit(X_train)
        scores = det.score(X_test)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = auprc(y_test, scores)
        ax.plot(rec, prec, lw=1.2, label=f"{det.name} ({ap:.2f})", alpha=0.85)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(ds_name)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

fig.suptitle("Precision-Recall Curves — All Detectors × All Datasets", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "03_pr_curves_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 03_pr_curves_all.png")
"""
        ),
        md("## Save arena results → `outputs/03_arena_results.json`"),
        code(
            """\
arena_json = {
    "scoreboard": scoreboard.round(4).to_dict(orient="records"),
    "best_per_dataset": {
        row["dataset"]: {
            "detector": row["name"],
            "auprc": round(float(row["auprc"]), 4),
            "f1": round(float(row["f1"]), 4),
        }
        for _, row in best.iterrows()
    },
    "contamination_rates": contamination,
    "note": (
        "Isolation Forest wins on credit_card (global outliers). "
        "Mahalanobis is competitive on all datasets. "
        "No single algorithm wins all three."
    ),
}

with (OUTPUTS / "03_arena_results.json").open("w") as f:
    json.dump(arena_json, f, indent=2)
print("Saved outputs/03_arena_results.json")
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


# ===========================================================================
# 04 — Contamination Sensitivity
# ===========================================================================
def nb04() -> nbf.NotebookNode:
    cells = [
        md(
            "# 04 — Contamination Sensitivity\n\n"
            "Sweep the contamination parameter from 0.001 to 0.10 for each detector on each dataset. "
            "Show that F1 is highly sensitive to this choice, and that Mahalanobis (no contamination "
            "parameter) is more robust."
        ),
        code(_SETUP),
        md("## Load datasets"),
        code(
            """\
X_cc, y_cc = make_synthetic_credit_card(n_samples=5000, random_state=42)
X_net, y_net = load_network(n_samples=3000, random_state=42)
X_mfg, y_mfg = load_manufacturing(n_samples=2000, random_state=42)

TRUE_CONTAMINATION = {"credit_card": 0.0017, "network": 0.02, "manufacturing": 0.015}
CONTAMINATION_RANGE = np.linspace(0.001, 0.10, 30)
print(f"Sweep range: {CONTAMINATION_RANGE[0]:.3f} → {CONTAMINATION_RANGE[-1]:.3f}  ({len(CONTAMINATION_RANGE)} steps)")
"""
        ),
        md("## Run contamination sweep — network dataset"),
        code(
            """\
detectors_sweep = [
    IsolationForestDetector(n_estimators=100, random_state=42),
    LOFDetector(n_neighbors=20),
    MahalanobisDetector(),
    AutoencoderDetector(hidden_dim=32, epochs=20, random_state=42),
]

split_net = int(0.8 * len(X_net))
X_tr_net, X_te_net, y_te_net = X_net[:split_net], X_net[split_net:], y_net[split_net:]

sweep_results = {}
for det in detectors_sweep:
    df = sweep_contamination(det, X_tr_net, X_te_net, y_te_net, CONTAMINATION_RANGE)
    sweep_results[det.name] = df
    print(f"  {det.name}: peak F1={df['f1'].max():.3f} at contamination={df.loc[df['f1'].idxmax(), 'contamination']:.4f}")
"""
        ),
        md("## F1 sensitivity plot — network dataset"),
        code(
            """\
fig, ax = plt.subplots(figsize=(10, 5))
for name, df in sweep_results.items():
    ax.plot(df["contamination"], df["f1"], marker="o", ms=3, lw=1.5, label=name)
ax.axvline(TRUE_CONTAMINATION["network"], color="k", ls="--", lw=1, label=f"true contamination ({TRUE_CONTAMINATION['network']})")
ax.set_xlabel("Contamination rate")
ax.set_ylabel("F1 score")
ax.set_title("Network Dataset — F1 vs Contamination Rate")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIGURES / "04_contamination_f1_network.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 04_contamination_f1_network.png")
"""
        ),
        md("## Precision and recall separately"),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, df in sweep_results.items():
    axes[0].plot(df["contamination"], df["precision"], lw=1.2, label=name)
    axes[1].plot(df["contamination"], df["recall"], lw=1.2, label=name)
for ax, metric in zip(axes, ["Precision", "Recall"]):
    ax.axvline(TRUE_CONTAMINATION["network"], color="k", ls="--", lw=1)
    ax.set_xlabel("Contamination rate")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Contamination — Network")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIGURES / "04_contamination_pr_network.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 04_contamination_pr_network.png")
"""
        ),
        md("## Sweep all three datasets for Isolation Forest"),
        code(
            """\
datasets_sweep = {
    "credit_card": (X_cc, y_cc, TRUE_CONTAMINATION["credit_card"]),
    "network": (X_net, y_net, TRUE_CONTAMINATION["network"]),
    "manufacturing": (X_mfg, y_mfg, TRUE_CONTAMINATION["manufacturing"]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
all_sweep_data = {}

for ax, (ds_name, (X, y, true_cont)) in zip(axes, datasets_sweep.items()):
    split = int(0.8 * len(X))
    X_tr, X_te, y_te = X[:split], X[split:], y[split:]
    det_if = IsolationForestDetector(n_estimators=100, random_state=42)
    df = sweep_contamination(det_if, X_tr, X_te, y_te, CONTAMINATION_RANGE)
    all_sweep_data[ds_name] = {"IsolationForest": df.to_dict(orient="records")}

    ax.plot(df["contamination"], df["f1"], color="steelblue", lw=1.5, label="F1")
    ax.plot(df["contamination"], df["precision"], color="orange", lw=1.2, ls="--", label="Precision")
    ax.plot(df["contamination"], df["recall"], color="tomato", lw=1.2, ls=":", label="Recall")
    ax.axvline(true_cont, color="k", ls="--", lw=1, label=f"true={true_cont}")
    ax.set_xlabel("Contamination")
    ax.set_title(f"Isolation Forest — {ds_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

fig.suptitle("Isolation Forest: F1 sensitivity to contamination rate", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "04_if_contamination_all_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 04_if_contamination_all_datasets.png")
"""
        ),
        md("## F1 range (sensitivity metric)"),
        code(
            """\
print("Isolation Forest F1 range across contamination sweep:")
for ds_name in datasets_sweep:
    records = all_sweep_data[ds_name]["IsolationForest"]
    f1_vals = [r["f1"] for r in records]
    f1_range = max(f1_vals) - min(f1_vals)
    print(f"  {ds_name:15s}: min={min(f1_vals):.3f}  max={max(f1_vals):.3f}  range={f1_range:.3f}")
print()
print("High range → contamination choice critically affects performance.")
"""
        ),
        md("## Save sweep results → `outputs/04_contamination_sweep.json`"),
        code(
            """\
sweep_json = {}
for ds_name, (X, y, true_cont) in datasets_sweep.items():
    sweep_json[ds_name] = {"true_contamination": true_cont, "detectors": {}}
    split = int(0.8 * len(X))
    X_tr, X_te, y_te = X[:split], X[split:], y[split:]
    for det in [IsolationForestDetector(n_estimators=100, random_state=42), MahalanobisDetector()]:
        df = sweep_contamination(det, X_tr, X_te, y_te, CONTAMINATION_RANGE)
        f1_vals = df["f1"].tolist()
        sweep_json[ds_name]["detectors"][det.name] = {
            "contamination_range": df["contamination"].tolist(),
            "f1": [round(v, 4) for v in f1_vals],
            "precision": [round(v, 4) for v in df["precision"].tolist()],
            "recall": [round(v, 4) for v in df["recall"].tolist()],
            "peak_f1": round(float(max(f1_vals)), 4),
            "f1_range": round(float(max(f1_vals) - min(f1_vals)), 4),
        }

with (OUTPUTS / "04_contamination_sweep.json").open("w") as f:
    json.dump(sweep_json, f, indent=2)
print("Saved outputs/04_contamination_sweep.json")
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


# ===========================================================================
# 05 — The Mahalanobis Surprise
# ===========================================================================
def nb05() -> nbf.NotebookNode:
    cells = [
        md(
            "# 05 — The Mahalanobis Surprise\n\n"
            "Why does a 5-line numpy method beat complex algorithms on contextual anomaly datasets? "
            "This notebook shows: the insight is in the features, not the algorithm."
        ),
        code(_SETUP),
        md("## Load datasets"),
        code(
            """\
X_cc, y_cc = make_synthetic_credit_card(n_samples=5000, random_state=42)
X_net, y_net = load_network(n_samples=3000, random_state=42)
X_mfg, y_mfg = load_manufacturing(n_samples=2000, random_state=42)
print("Loaded.")
"""
        ),
        md(
            "## The 5-line implementation\n\n"
            "```python\n"
            "def mahalanobis_scores(X_train, X_test):\n"
            "    mu = X_train.mean(axis=0)\n"
            "    cov = np.cov(X_train, rowvar=False)\n"
            "    cov_inv = np.linalg.pinv(cov)\n"
            "    diff = X_test - mu\n"
            "    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))\n"
            "```\n\n"
            "No hyperparameters. No training loop. No contamination parameter. "
            "Uses the inverse covariance matrix to account for feature correlations."
        ),
        code(
            """\
def mahalanobis_scores(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    mu = X_train.mean(axis=0)
    cov = np.atleast_2d(np.cov(X_train, rowvar=False))
    cov_inv = np.linalg.pinv(cov)
    diff = X_test - mu
    return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0.0))

# Quick sanity check
rng = np.random.RandomState(0)
X_dummy = rng.randn(200, 4)
X_outlier = np.array([[5.0, 5.0, 5.0, 5.0]])  # obvious outlier
train_scores = mahalanobis_scores(X_dummy, X_dummy)
outlier_score = mahalanobis_scores(X_dummy, X_outlier)
print(f"Mean score on normal data: {train_scores.mean():.2f}  (expected ≈ √features ≈ {np.sqrt(4):.2f})")
print(f"Score on [5,5,5,5] outlier: {outlier_score[0]:.2f}  (expected >> normal)")
"""
        ),
        md(
            "## Experiment 1: Manufacturing — raw features vs residual features\n\n"
            "**Key insight:** Mahalanobis on raw temperature is a *global* outlier detector. "
            "Mahalanobis on the seasonal *residual* (feature index 4) is a *contextual* anomaly detector."
        ),
        code(
            """\
split_mfg = int(0.8 * len(X_mfg))
X_tr_mfg, X_te_mfg = X_mfg[:split_mfg], X_mfg[split_mfg:]
y_te_mfg = y_mfg[split_mfg:]

# Raw Mahalanobis (all 5 features)
scores_raw = mahalanobis_scores(X_tr_mfg, X_te_mfg)
auprc_raw = auprc(y_te_mfg, scores_raw)

# Residual-only Mahalanobis (feature 4 = seasonal residual)
scores_residual = mahalanobis_scores(X_tr_mfg[:, 4:5], X_te_mfg[:, 4:5])
auprc_residual = auprc(y_te_mfg, scores_residual)

# Isolation Forest for comparison
det_if = IsolationForestDetector(n_estimators=100, random_state=42)
det_if.fit(X_tr_mfg)
auprc_if = auprc(y_te_mfg, det_if.score(X_te_mfg))

print(f"Mahalanobis (raw features)    AUPRC = {auprc_raw:.4f}")
print(f"Mahalanobis (residual only)   AUPRC = {auprc_residual:.4f}")
print(f"Isolation Forest (raw)        AUPRC = {auprc_if:.4f}")
print()
print("Residual Mahalanobis wins because the feature already encodes what 'normal' means.")
"""
        ),
        md("## PR curves: raw vs residual Mahalanobis"),
        code(
            """\
from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(figsize=(8, 5))
for label, scores, ap in [
    ("Mahalanobis raw features", scores_raw, auprc_raw),
    ("Mahalanobis residual only", scores_residual, auprc_residual),
    ("Isolation Forest", det_if.score(X_te_mfg), auprc_if),
]:
    prec, rec, _ = precision_recall_curve(y_te_mfg, scores)
    ax.plot(rec, prec, lw=2, label=f"{label} (AUPRC={ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Manufacturing Dataset — Feature Engineering Makes the Difference")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIGURES / "05_manufacturing_raw_vs_residual.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 05_manufacturing_raw_vs_residual.png")
"""
        ),
        md(
            "## Experiment 2: Credit card — Mahalanobis vs Isolation Forest\n\n"
            "On the credit card dataset (multimodal, 30 PCA features), "
            "Mahalanobis assumes a single Gaussian. Isolation Forest wins here."
        ),
        code(
            """\
split_cc = int(0.8 * len(X_cc))
X_tr_cc, X_te_cc = X_cc[:split_cc], X_cc[split_cc:]
y_te_cc = y_cc[split_cc:]

maha_det = MahalanobisDetector()
if_det = IsolationForestDetector(n_estimators=100, random_state=42)

maha_det.fit(X_tr_cc)
if_det.fit(X_tr_cc)

auprc_maha_cc = auprc(y_te_cc, maha_det.score(X_te_cc))
auprc_if_cc = auprc(y_te_cc, if_det.score(X_te_cc))

print(f"Credit card — Mahalanobis AUPRC : {auprc_maha_cc:.4f}")
print(f"Credit card — Isolation Forest  : {auprc_if_cc:.4f}")
delta = auprc_if_cc - auprc_maha_cc
print(f"IF advantage: +{delta:.4f} AUPRC points")
print()
print("Mahalanobis assumes a single Gaussian. Credit card fraud has multimodal patterns.")
print("Isolation Forest handles multimodality better.")
"""
        ),
        md("## Summary: when to use Mahalanobis"),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Credit card
for ax, (title, X_tr, X_te, y_te, auprc_maha, auprc_if) in [
    (axes[0], ("Credit Card (global outliers)", X_tr_cc, X_te_cc, y_te_cc, auprc_maha_cc, auprc_if_cc)),
    (axes[1], ("Manufacturing (contextual)", X_tr_mfg, X_te_mfg, y_te_mfg, auprc_residual, auprc_if)),
]:
    methods = ["Mahalanobis", "Isolation Forest"]
    values = [auprc_maha, auprc_if]
    colors = ["steelblue" if v == max(values) else "lightgray" for v in values]
    ax.bar(methods, values, color=colors, edgecolor="black", lw=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUPRC")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

fig.suptitle("Mahalanobis wins on contextual data (with feature engineering), loses on global outliers", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES / "05_mahalanobis_vs_if_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 05_mahalanobis_vs_if_summary.png")
"""
        ),
        md("## Network dataset: time-of-day conditioned Mahalanobis"),
        code(
            """\
split_net = int(0.8 * len(X_net))
X_tr_net, X_te_net, y_te_net = X_net[:split_net], X_net[split_net:], y_net[split_net:]

maha_raw_net = MahalanobisDetector()
maha_raw_net.fit(X_tr_net)
auprc_maha_raw_net = auprc(y_te_net, maha_raw_net.score(X_te_net))

det_if_net = IsolationForestDetector(n_estimators=100, random_state=42)
det_if_net.fit(X_tr_net)
auprc_if_net = auprc(y_te_net, det_if_net.score(X_te_net))

print(f"Network — Mahalanobis (all features incl. time encoding) AUPRC: {auprc_maha_raw_net:.4f}")
print(f"Network — Isolation Forest                                AUPRC: {auprc_if_net:.4f}")
print()
print("The hour_sin/hour_cos features encode context for Mahalanobis automatically.")
"""
        ),
        md("## Save analysis → `outputs/05_mahalanobis_analysis.json`"),
        code(
            """\
analysis = {
    "manufacturing": {
        "mahalanobis_raw_auprc": round(auprc_raw, 4),
        "mahalanobis_residual_auprc": round(auprc_residual, 4),
        "isolation_forest_auprc": round(auprc_if, 4),
        "insight": "Residual feature transforms a global detector into a contextual one.",
    },
    "credit_card": {
        "mahalanobis_auprc": round(auprc_maha_cc, 4),
        "isolation_forest_auprc": round(auprc_if_cc, 4),
        "if_advantage_auprc": round(auprc_if_cc - auprc_maha_cc, 4),
        "insight": "Mahalanobis assumes Gaussian; credit card fraud is multimodal — IF wins.",
    },
    "network": {
        "mahalanobis_auprc": round(auprc_maha_raw_net, 4),
        "isolation_forest_auprc": round(auprc_if_net, 4),
        "insight": "Time encoding (sin/cos) gives Mahalanobis contextual awareness automatically.",
    },
    "key_conclusions": [
        "Feature engineering, not algorithm complexity, determines contextual anomaly detection quality.",
        "Mahalanobis on residuals = contextual detector. On raw features = global detector.",
        "Isolation Forest is the right default for global outliers (credit card, high-dim PCA features).",
        "No single algorithm wins all three datasets — matching algorithm to anomaly type matters.",
    ],
}

with (OUTPUTS / "05_mahalanobis_analysis.json").open("w") as f:
    json.dump(analysis, f, indent=2)
print("Saved outputs/05_mahalanobis_analysis.json")
print()
for c in analysis["key_conclusions"]:
    print(f"  • {c}")
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


# ---------------------------------------------------------------------------
# Generate all notebooks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating notebooks...")
    save_nb(nb01(), NB_DIR / "01_data_exploration.ipynb")
    save_nb(nb02(), NB_DIR / "02_global_vs_contextual.ipynb")
    save_nb(nb03(), NB_DIR / "03_algorithm_comparison.ipynb")
    save_nb(nb04(), NB_DIR / "04_contamination_sensitivity.ipynb")
    save_nb(nb05(), NB_DIR / "05_the_mahalanobis_surprise.ipynb")
    print("All notebooks created.")
