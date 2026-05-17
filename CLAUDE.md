# anomaly_detection_arena

Head-to-head comparison of 6 anomaly detection algorithms on 3 datasets.
Thesis: Isolation Forest misses contextual anomalies that 5-line Mahalanobis catches.

## Stack

- Python 3.11+, `uv` (no pip ever), `ruff` + `ty` linting, `pytest` with coverage
- sklearn, scipy, numpy, torch (CPU-only, no CUDA), matplotlib, seaborn, streamlit
- Constraint: 2-CPU / 8 GB RAM GitHub Codespace

## Repository Layout

```
src/
  config.py
  datasets/        credit_card.py · network.py · manufacturing.py
  detectors/       isolation_forest.py · lof.py · one_class_svm.py
                   dbscan_detector.py · mahalanobis.py · autoencoder.py
  evaluation/      metrics.py · comparison.py · contamination_sweep.py
  visualisation.py
app/streamlit_app.py
tests/             test_datasets.py · test_detectors.py · test_metrics.py
                   test_mahalanobis.py · test_comparison.py
notebooks/         01–05 .ipynb
config/settings.yaml
data/              credit card CSV cached here after `make data`
outputs/figures/
```

## Commands

```bash
make setup       # first-time: uv sync + ipykernel
make data        # download + cache credit card dataset
make lint        # ruff format + check + ty typecheck
make test        # pytest --cov=src --cov-fail-under=80  ← enforces 80% coverage
make test-cov    # same + HTML report in htmlcov/
make notebooks   # execute all notebooks (requires make data first)
make run         # streamlit app on :8501
make lab         # JupyterLab on :8888
make ci          # sync + lint + test
```

## Testing Requirements

- `make test` runs coverage and **fails if coverage < 80%**
- Never skip coverage — CI requires it
- Tests must be fast: use small synthetic fixtures, not full datasets
- Credit card tests: use 100-row subset; network/manufacturing: 200-row subset

## Key Concepts

**Global outliers** — far from everything in feature space (credit card fraud).
Isolation Forest, One-Class SVM, LOF handle these well.

**Contextual anomalies** — normal absolute value, abnormal relative to context
(3 AM network scan at normal request rate; 45 °C during overnight cooling cycle).
Global detectors miss these. Mahalanobis on *residuals* catches them.

**The Mahalanobis insight:**
- `mahalanobis(raw_features)` → global outlier detector
- `mahalanobis(seasonal_residuals)` → contextual anomaly detector
- Feature engineering does the work, not the algorithm

**Evaluation — never use accuracy:**
- Metric: precision, recall, F1 at the true contamination rate; AUPRC
- At 0.17% fraud rate, "predict normal always" = 99.83% accuracy — useless
- AUPRC is the honest metric for imbalanced anomaly detection

## Algorithms

| # | Name | Key strength | Key weakness |
|---|------|-------------|-------------|
| 1 | Isolation Forest | Fast, high-dim | Misses contextual |
| 2 | LOF | Local density deviations | Slow, k-sensitive |
| 3 | One-Class SVM | Tight boundary | Slow; subsample to 10K rows |
| 4 | DBSCAN noise | No contamination param | eps/min_samples sensitive |
| 5 | Mahalanobis | No hyperparams; 5 lines | Assumes Gaussian |
| 6 | Autoencoder | Nonlinear patterns | Slow on CPU; threshold choice |

## Datasets

| # | Name | Rows | Anomaly rate | Type |
|---|------|------|-------------|------|
| 1 | Credit card fraud | 284,807 | 0.17% | Global outliers |
| 2 | Synthetic network intrusion | 10,000 | 2% | Contextual (time-of-day) |
| 3 | Synthetic manufacturing sensor | 8,000 | 1.5% | Contextual (seasonal) |

Expected arena results:
- Credit card: Isolation Forest / Autoencoder lead; Mahalanobis competitive
- Network: Mahalanobis (time-conditioned) leads; Isolation Forest 4th–5th
- Manufacturing: Mahalanobis (seasonal residuals) leads; Isolation Forest ~0

## Coding Conventions

- `ruff` line-length 99, target py311
- `ty` for type checking (`src/` only)
- All detectors implement a common interface: `fit(X_train)` → `score(X_test)` → `predict(X_test, contamination)`
- `n_jobs=2` everywhere (Codespace constraint)
- Autoencoder: 2 layers, 32 hidden units, CPU-only, trains < 60 s on 10K rows
- One-Class SVM: always subsample to 10,000 rows; document this

## Config

`config/settings.yaml` holds contamination rates, thresholds, model params.
`src/config.py` loads it via `pydantic-settings`. Never hardcode hyperparams.
