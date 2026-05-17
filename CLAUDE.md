# anomaly_detection_arena

Head-to-head comparison of 6 anomaly detection algorithms on 3 datasets.
**Thesis:** Isolation Forest misses contextual anomalies that a 5-line Mahalanobis method catches —
and the reason is feature engineering, not algorithm complexity.

---

## Stack

- Python 3.11+, `uv` (no `pip` ever), `ruff` + `ty` linting, `pytest` with coverage
- sklearn, scipy, numpy, torch (CPU-only, no CUDA), matplotlib, streamlit
- Constraint: 2-CPU / 8 GB RAM GitHub Codespace

---

## Repository Layout

```
src/
  config.py              pydantic model, loads config/settings.yaml
  datasets/              credit_card.py · network.py · manufacturing.py
  detectors/             base.py · isolation_forest.py · lof.py · one_class_svm.py
                         dbscan_detector.py · mahalanobis.py · autoencoder.py
  evaluation/            metrics.py · comparison.py · contamination_sweep.py
  visualisation.py       4 plot functions returning Figure
app/
  streamlit_app.py
notebooks/               01–05 .ipynb  (executed, outputs embedded)
scripts/
  generate_notebooks.py  regenerates all notebooks from source
tests/                   test_datasets · test_detectors · test_metrics
                         test_mahalanobis · test_comparison  (144 tests, 94% coverage)
config/settings.yaml
data/                    credit card CSV cached here after `make data`
outputs/
  figures/               14 PNG figures (150 dpi)
  *.json                 5 results JSON files (one per notebook)
```

---

## Commands

```bash
make setup       # uv sync --all-extras + ipykernel
make data        # download + cache real credit card CSV
make lint        # ruff format → ruff check --fix → ty check src/
make test        # pytest --cov=src --cov-fail-under=80  (currently 94%)
make test-cov    # same + htmlcov/ HTML report
make notebooks   # execute all 5 notebooks in-place (--inplace)
make run         # streamlit run app/streamlit_app.py --server.port 8501
make lab         # JupyterLab on :8888
make ci          # sync + lint + test
make dev         # lint + test  (fast loop)
```

---

## Testing Rules

- `make test` enforces **≥ 80% coverage** (`--cov-fail-under=80`)
- Never import the real credit card CSV in tests — use `make_synthetic_credit_card()`
- Keep test fixtures small: ≤ 100 rows for detectors, ≤ 300 for datasets
- Tests must run in < 20 s total. Use `epochs=3` for the autoencoder in tests.

---

## Detector Interface

All detectors extend `src/detectors/base.py::BaseDetector`:

```python
class BaseDetector(ABC):
    name: str                                        # human-readable, used in scoreboard

    def fit(self, X: np.ndarray) -> "BaseDetector"
    def score(self, X: np.ndarray) -> np.ndarray    # float64, higher = more anomalous
    def predict(self, X, contamination) -> np.ndarray  # int, 1=anomaly; default impl in base
```

`score()` returns raw anomaly scores. `predict()` thresholds at `np.quantile(scores, 1-contamination)`.
**Never bake `contamination` into `fit()` or `score()`.** The Streamlit app caches scores and
applies the threshold dynamically — if contamination leaks into score computation the slider breaks.

---

## Key Concepts

**Global outlier** — far from everything in feature space. Isolation Forest, LOF, One-Class SVM
handle these by design.

**Contextual anomaly** — normal absolute value, abnormal relative to context:
- Network: high request rate at 3 AM (same rate is normal at noon)
- Manufacturing: 45 °C during overnight cooling (normal during peak production)

**The Mahalanobis insight:**
```python
mahalanobis(raw_temp)      # global detector — misses contextual anomalies
mahalanobis(residual)      # contextual detector — the feature does the work
```
Feature engineering (seasonal residuals, time sin/cos encoding) is the real lever.

**Evaluation — never accuracy:**
- 0.17% fraud rate → "predict normal always" = 99.83% accuracy, useless
- Use: precision, recall, F1 at the true contamination rate; AUPRC
- `src/evaluation/metrics.py::precision_recall_f1_at_contamination` + `auprc`

---

## Algorithms

| # | Class | Key param | Subsample? |
|---|-------|-----------|------------|
| 1 | `IsolationForestDetector` | `n_estimators=100`, `n_jobs=2` | — |
| 2 | `LOFDetector` | `n_neighbors=20`, `n_jobs=2`, `novelty=True` | — |
| 3 | `OneClassSVMDetector` | `nu=0.05` | ≤ 10K rows at fit |
| 4 | `DBSCANDetector` | `eps=0.5`, `min_samples=5` | — |
| 5 | `MahalanobisDetector` | none (`pinv` for near-singular cov) | — |
| 6 | `AutoencoderDetector` | `hidden_dim=32`, `epochs=50`, CPU-only | — |

---

## Datasets

| Name | Loader | Rows | Anomaly rate | Anomaly type |
|------|--------|------|-------------|--------------|
| Credit card | `load_credit_card()` / `make_synthetic_credit_card()` | 284K / synthetic | 0.17% | Global outlier |
| Network | `load_network(n_samples=10000)` | synthetic | 2% | Contextual (time-of-day) |
| Manufacturing | `load_manufacturing(n_samples=8000)` | synthetic | 1.5% | Contextual (seasonal) |

Network features: `hour_sin, hour_cos, request_rate, session_duration, subnet_score`.
Manufacturing features: `raw_temp, hour_sin, hour_cos, seasonal_expected, residual`.

---

## Visualisation

`src/visualisation.py` exposes four functions, all return `matplotlib.figure.Figure`:

| Function | What it shows | Dynamic? |
|----------|--------------|----------|
| `plot_score_map(X, scores, contamination, y_true)` | PCA scatter; red=flagged, blue=normal, ✕=true anomaly | Yes — re-drawn per contamination |
| `plot_precision_recall_curve(y_true, scores, contamination)` | PR curve + red dot at operating point | Dot moves with contamination |
| `plot_arena_scoreboard(df, metric)` | Heatmap detectors × datasets | — |
| `plot_contamination_sweep(df)` | Precision/recall/F1 vs contamination rate | — |

---

## Streamlit App Caching Rule

```python
@st.cache_data
def get_scores(detector_name, dataset_name):   # ← NO contamination here
    ...
    return scores, y_test, X_test
```

Contamination must **never** enter the cache key. Scores are fixed after fitting; the threshold is
applied outside the cache so the slider re-renders immediately without re-fitting.

---

## Notebooks

Each notebook saves figures to `outputs/figures/` and results to `outputs/*.json`.
Notebooks are executed with `--inplace` so outputs are embedded in the `.ipynb` file directly.
Re-generate from source without outputs: `uv run python scripts/generate_notebooks.py`.

| Notebook | Saves |
|----------|-------|
| `01_data_exploration` | `01_dataset_summary.json`, 3 figures |
| `02_global_vs_contextual` | `02_global_vs_contextual.json`, 3 figures |
| `03_algorithm_comparison` | `03_arena_results.json`, 3 figures |
| `04_contamination_sensitivity` | `04_contamination_sweep.json`, 3 figures |
| `05_the_mahalanobis_surprise` | `05_mahalanobis_analysis.json`, 2 figures |

---

## Coding Conventions

- `ruff` line-length 99, target py311. Rules: `E,F,W,I,UP,N,B,A,SIM,PTH`. N803/N806 ignored
  (ML convention: `X_train`, `X_test` are acceptable variable names).
- `ty check src/` — type-check library code only, not tests or app.
- Use `pathlib.Path` everywhere; never `os.path`.
- Modern type syntax: `list[int]`, `dict[str, float]`, `X | None` (not `Optional`).
- No hardcoded hyperparams — everything goes through `config/settings.yaml`.
- `n_jobs=2` for parallelisable sklearn estimators (Codespace limit).
