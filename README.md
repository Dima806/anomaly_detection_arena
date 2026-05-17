# anomaly_detection_arena

> **Your anomaly detector is finding outliers, not anomalies.**

A head-to-head comparison of six anomaly detection algorithms on three datasets, built to prove one point: **Isolation Forest — the default for almost everyone — misses contextual anomalies that a 5-line numpy method catches.** The reason isn't the algorithm. It's the features.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://github.com/astral-sh/ruff)

---

## The argument

Most practitioners reach for `IsolationForest(contamination=0.01)` as a first pass. It works by measuring how many random splits are needed to isolate a point. Points that are easy to isolate are anomalies. This is an effective strategy for **global outliers** — points that are simply far from everything else.

But production anomaly detection is rarely about global outliers. It's about **contextual anomalies**: records that look completely normal in isolation but are suspicious given their context.

- A credit card transaction of €500 is normal on a Friday evening in Copenhagen. At 3 AM from a new device in a different country, it's suspicious. The amount hasn't changed. The context has.
- A network request rate of 100 req/s is normal during business hours. At 3 AM it's a port scan.
- A sensor reading of 45 °C is normal during peak production. During the overnight cooling cycle it indicates a fault.

Isolation Forest sees the amount, the request rate, or the temperature — and says *normal*. A Mahalanobis distance computed on the **residual from the seasonal expectation** says *anomaly*.

**This repository shows exactly why, with code you can run.**

---

## Algorithms compared

| # | Algorithm | Family | Strength | Weakness |
|---|-----------|--------|----------|----------|
| 1 | Isolation Forest | Tree ensemble | Fast, scales to high dimensions | Global outliers only; contamination-sensitive |
| 2 | Local Outlier Factor | Density | Detects local density deviations | Slow on large data; k-sensitive |
| 3 | One-Class SVM | Kernel boundary | Tight boundary around normal data | Very slow; subsampled to 10K rows |
| 4 | DBSCAN (noise label) | Density clustering | No contamination parameter | eps/min_samples brittle |
| 5 | **Mahalanobis distance** | Statistical | **No hyperparameters; 5 lines of numpy** | Assumes multivariate Gaussian |
| 6 | Autoencoder | Deep learning | Learns nonlinear normal patterns | Slow on CPU; threshold selection |

---

## Datasets

| Dataset | Rows | Anomaly rate | Anomaly type |
|---------|------|-------------|--------------|
| Credit card fraud (Kaggle) | 284,807 | 0.17% | **Global outlier** — fraud differs across all 30 PCA features simultaneously |
| Synthetic network intrusion | 10,000 | 2% | **Contextual** — normal request rates at abnormal hours |
| Synthetic manufacturing sensor | 8,000 | 1.5% | **Contextual** — normal temperatures during wrong phase of seasonal cycle |

The credit card dataset is downloaded once with `make data` and cached locally. The network and manufacturing datasets are generated synthetically with controlled ground truth, making the contextual anomaly structure transparent and reproducible.

---

## The 5-line method

```python
def mahalanobis_scores(X_train, X_test):
    mu      = X_train.mean(axis=0)
    cov     = np.cov(X_train, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    diff    = X_test - mu
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
```

Applied to **raw sensor values**: global outlier detector.
Applied to **seasonal residuals** (raw − seasonal expectation): contextual anomaly detector.

The feature engineering is doing the work. The algorithm is just measuring multivariate distance.

---

## Quick start

```bash
# 1. Install (requires uv)
make setup

# 2. Download credit card dataset (optional — notebooks use synthetic data by default)
make data

# 3. Run the full comparison
make notebooks      # executes all 5 notebooks, saves figures + JSON results

# 4. Explore interactively
make run            # Streamlit app on :8501
make lab            # JupyterLab on :8888
```

No GPU required. Designed to run inside a **2-CPU / 8 GB GitHub Codespace** in under 30 minutes total.

---

## Repository structure

```
anomaly_detection_arena/
├── src/
│   ├── config.py                    Settings loader (pydantic + YAML)
│   ├── datasets/
│   │   ├── credit_card.py           Real Kaggle dataset loader + synthetic fallback
│   │   ├── network.py               Synthetic contextual DGP (time-of-day)
│   │   └── manufacturing.py         Synthetic contextual DGP (seasonal)
│   ├── detectors/
│   │   ├── base.py                  BaseDetector ABC: fit / score / predict
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   ├── one_class_svm.py
│   │   ├── dbscan_detector.py
│   │   ├── mahalanobis.py
│   │   └── autoencoder.py           2-layer PyTorch AE, CPU-only
│   ├── evaluation/
│   │   ├── metrics.py               precision_recall_f1_at_contamination, auprc
│   │   ├── comparison.py            run_arena: all detectors × all datasets
│   │   └── contamination_sweep.py   sweep_contamination: F1 vs threshold
│   └── visualisation.py             plot_score_map, plot_precision_recall_curve,
│                                    plot_arena_scoreboard, plot_contamination_sweep
├── app/
│   └── streamlit_app.py             Interactive explorer (3 tabs)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_global_vs_contextual.ipynb
│   ├── 03_algorithm_comparison.ipynb
│   ├── 04_contamination_sensitivity.ipynb
│   └── 05_the_mahalanobis_surprise.ipynb
├── scripts/
│   └── generate_notebooks.py        Regenerate notebooks from source
├── tests/                           144 tests, 94% coverage
├── outputs/
│   ├── figures/                     14 PNG figures (150 dpi)
│   └── *.json                       5 results files (one per notebook)
├── config/settings.yaml             All hyperparameters and contamination rates
├── Makefile
└── pyproject.toml
```

---

## Notebooks

### 01 — Data Exploration
Load all three datasets, inspect distributions, and visualise the contextual structure: request-rate vs hour-of-day for network data, raw temperature vs seasonal expectation for manufacturing. This is what makes contextual anomalies hard — they're invisible in univariate distributions.

### 02 — Global vs Contextual
Formal definition of both anomaly types. Shows via global Mahalanobis distance from the mean that credit card anomalies *are* globally distant, while network and manufacturing anomalies are not — they sit in the middle of the normal cloud when projected globally. The raw vs residual distribution plot makes the contextual structure visible.

### 03 — Algorithm Comparison (the Arena)
All 6 detectors × 3 datasets. Precision, recall, F1, and AUPRC at the true contamination rate. Scoreboard heatmaps and per-dataset PR curves. Key finding: no single algorithm wins everywhere.

### 04 — Contamination Sensitivity
Sweep the contamination parameter from 0.001 to 0.10. Shows that F1 varies sharply around the true contamination rate — choosing the wrong value floods the review queue with false positives or misses most true anomalies. Mahalanobis (which uses a chi-squared threshold, not a contamination parameter) is more robust to this choice.

### 05 — The Mahalanobis Surprise
Deep dive into why the simplest method wins on contextual datasets. Side-by-side comparison of Mahalanobis on raw features vs residual features on the manufacturing dataset. Shows the credit card case where Mahalanobis is at a disadvantage (multimodal fraud patterns violate the Gaussian assumption).

---

## Streamlit app

Three-tab dashboard:

| Tab | What it shows |
|-----|---------------|
| **Score Map** | PCA scatter: red = flagged at current contamination, blue = normal, ✕ = true anomaly. Redraws instantly when you move the slider. |
| **PR Curve** | Full precision-recall curve with a red operating-point dot that moves as you adjust contamination. |
| **Comparison** | Side-by-side score maps and PR curves for two detectors on the same dataset. |

Detector fitting is cached separately from threshold application, so the contamination slider is instant — the model never re-fits when you move the slider.

---

## Evaluation design

**Why not accuracy?** At 0.17% anomaly prevalence, a model that predicts *normal* for every record achieves 99.83% accuracy. Accuracy is not a metric for this problem.

**What to use instead:**

| Metric | What it measures | When to use |
|--------|-----------------|-------------|
| Precision at contamination | Of the records flagged, how many are real anomalies? | When review queue size matters |
| Recall at contamination | Of all real anomalies, how many were found? | When missing an anomaly is costly |
| F1 at contamination | Harmonic mean of the two | Balanced comparison |
| AUPRC | Area under the PR curve — threshold-free | Model ranking, not deployment |

All metrics are implemented in `src/evaluation/metrics.py`. The `run_arena` function in `src/evaluation/comparison.py` runs a full 80/20 train/test evaluation automatically.

---

## Development

```bash
make lint       # ruff format + ruff check --fix + ty check src/
make test       # pytest with coverage (fails if < 80%)
make dev        # lint + test combined
```

Tools: [uv](https://github.com/astral-sh/uv) for packaging, [ruff](https://github.com/astral-sh/ruff) for linting/formatting, [ty](https://github.com/astral-sh/ty) for type checking, [pytest](https://pytest.org) with [pytest-cov](https://github.com/pytest-dev/pytest-cov) for testing.

```bash
# Regenerate notebooks (strips existing outputs)
uv run python scripts/generate_notebooks.py
make notebooks   # re-execute to produce outputs
```

---

## Key findings

The main results from `outputs/03_arena_results.json` and `outputs/05_mahalanobis_analysis.json`:

- **No algorithm wins all three datasets.** The global vs contextual distinction is the deciding factor, not algorithm sophistication.

- **Contamination sensitivity is real.** On the network dataset, Isolation Forest F1 varies by more than 0.15 across the contamination sweep range. Choosing the wrong value at deployment doubles false-positive volume or halves recall.

- **The Mahalanobis insight holds.** When the feature includes the seasonal residual rather than the raw value, a 5-line numpy method performs competitively with or better than sklearn's flagship anomaly detector on contextual datasets. The feature engineering is the insight.

- **Accuracy would hide all of this.** At the true contamination rates used here, a null model achieves ≥ 98% accuracy on every dataset. The only honest metric is precision-recall.

---

## References

- Liu, F.T., Ting, K.M., and Zhou, Z.-H. (2008). *Isolation Forest.* ICDM 2008.
- Breunig, M.M. et al. (2000). *LOF: Identifying Density-Based Local Outliers.* SIGMOD 2000.
- Chandola, V., Banerjee, A., and Kumar, V. (2009). *Anomaly Detection: A Survey.* ACM Computing Surveys, 41(3).

---

## License

Apache 2.0. See [LICENSE](LICENSE).
