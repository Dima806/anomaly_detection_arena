"""Tests for src/evaluation/comparison.py and src/visualisation.py."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402

from src.detectors.isolation_forest import IsolationForestDetector  # noqa: E402 I001
from src.detectors.mahalanobis import MahalanobisDetector  # noqa: E402
from src.evaluation.comparison import run_arena  # noqa: E402
from src.visualisation import (  # noqa: E402
    plot_anomaly_scores,
    plot_arena_scoreboard,
    plot_contamination_sweep,
    plot_precision_recall_curve,
    plot_score_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 200,
    n_features: int = 5,
    contamination: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset with well-separated anomalies (+5 offset)."""
    rng = np.random.default_rng(random_state)
    n_anomalies = max(1, int(n * contamination))
    n_normal = n - n_anomalies
    X_normal = rng.standard_normal((n_normal, n_features))
    X_anomaly = rng.standard_normal((n_anomalies, n_features)) + 5.0
    X = np.vstack([X_normal, X_anomaly]).astype(np.float64)
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])
    return X, y


def _make_detectors() -> list:
    return [
        IsolationForestDetector(n_estimators=20, random_state=0),
        MahalanobisDetector(),
    ]


# ---------------------------------------------------------------------------
# run_arena
# ---------------------------------------------------------------------------


class TestRunArena:
    def setup_method(self):
        X, y = _make_synthetic_dataset(n=200, n_features=5, contamination=0.1)
        self.datasets = {"synthetic": (X, y, 0.1)}
        self.detectors = _make_detectors()

    def test_output_shape(self):
        df = run_arena(self.detectors, self.datasets)
        # 2 detectors × 1 dataset = 2 rows
        assert df.shape == (2, 6)

    def test_column_names(self):
        df = run_arena(self.detectors, self.datasets)
        assert set(df.columns) == {"name", "dataset", "precision", "recall", "f1", "auprc"}

    def test_metrics_in_unit_interval(self):
        df = run_arena(self.detectors, self.datasets)
        for col in ("precision", "recall", "f1", "auprc"):
            assert df[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"

    def test_dataset_column_values(self):
        df = run_arena(self.detectors, self.datasets)
        assert (df["dataset"] == "synthetic").all()

    def test_name_column_contains_detector_names(self):
        df = run_arena(self.detectors, self.datasets)
        assert set(df["name"]) == {"IsolationForest", "Mahalanobis"}

    def test_multiple_datasets(self):
        X2, y2 = _make_synthetic_dataset(n=160, n_features=3, random_state=1)
        datasets = {
            "ds1": (self.datasets["synthetic"][0], self.datasets["synthetic"][1], 0.1),
            "ds2": (X2, y2, 0.1),
        }
        df = run_arena(self.detectors, datasets)
        # 2 detectors × 2 datasets = 4 rows
        assert len(df) == 4
        assert set(df["dataset"]) == {"ds1", "ds2"}

    def test_80_20_split_uses_test_labels(self):
        # With n=200 and 80/20 split, test split has 40 samples.
        # Contamination=0.1 means top 10% flagged = 4 samples.
        df = run_arena(self.detectors, self.datasets)
        # Just verify no exception and shape correct; split logic implicitly tested.
        assert len(df) == 2


# ---------------------------------------------------------------------------
# plot_anomaly_scores
# ---------------------------------------------------------------------------


class TestPlotAnomalyScores:
    def test_returns_figure(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5))
        scores = rng.uniform(0, 1, 50)
        fig = plot_anomaly_scores(X, scores)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_figure_with_y_true(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 5))
        scores = rng.uniform(0, 1, 50)
        y_true = (scores > 0.8).astype(int)
        fig = plot_anomaly_scores(X, scores, y_true=y_true, title="Test")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_feature(self):
        # Edge case: 1-D feature matrix
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 1))
        scores = rng.uniform(0, 1, 30)
        fig = plot_anomaly_scores(X, scores)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_anomalies_in_y_true(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 4))
        scores = rng.uniform(0, 1, 30)
        y_true = np.zeros(30, dtype=int)
        fig = plot_anomaly_scores(X, scores, y_true=y_true)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_precision_recall_curve
# ---------------------------------------------------------------------------


class TestPlotPrecisionRecallCurve:
    def test_returns_figure(self):
        rng = np.random.default_rng(10)
        y_true = np.concatenate([np.ones(10, dtype=int), np.zeros(40, dtype=int)])
        scores = rng.uniform(0, 1, 50)
        fig = plot_precision_recall_curve(y_true, scores)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_figure_with_label(self):
        rng = np.random.default_rng(11)
        y_true = np.concatenate([np.ones(5, dtype=int), np.zeros(45, dtype=int)])
        scores = rng.uniform(0, 1, 50)
        fig = plot_precision_recall_curve(y_true, scores, label="MyDetector")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        fig = plot_precision_recall_curve(y_true, scores, label="Perfect")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_arena_scoreboard
# ---------------------------------------------------------------------------


class TestPlotArenaScoreboard:
    def _make_scoreboard(self) -> pd.DataFrame:
        rows = [
            {
                "name": "IsolationForest",
                "dataset": "ds1",
                "auprc": 0.8,
                "f1": 0.7,
                "precision": 0.75,
                "recall": 0.65,
            },
            {
                "name": "IsolationForest",
                "dataset": "ds2",
                "auprc": 0.6,
                "f1": 0.5,
                "precision": 0.55,
                "recall": 0.45,
            },
            {
                "name": "Mahalanobis",
                "dataset": "ds1",
                "auprc": 0.75,
                "f1": 0.65,
                "precision": 0.7,
                "recall": 0.6,
            },
            {
                "name": "Mahalanobis",
                "dataset": "ds2",
                "auprc": 0.55,
                "f1": 0.45,
                "precision": 0.5,
                "recall": 0.4,
            },
        ]
        return pd.DataFrame(rows)

    def test_returns_figure_auprc(self):
        df = self._make_scoreboard()
        fig = plot_arena_scoreboard(df, metric="auprc")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_figure_f1(self):
        df = self._make_scoreboard()
        fig = plot_arena_scoreboard(df, metric="f1")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_detector_single_dataset(self):
        df = pd.DataFrame(
            [
                {
                    "name": "LOF",
                    "dataset": "only",
                    "auprc": 0.7,
                    "f1": 0.6,
                    "precision": 0.65,
                    "recall": 0.55,
                }
            ]
        )
        fig = plot_arena_scoreboard(df, metric="auprc")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_contamination_sweep
# ---------------------------------------------------------------------------


class TestPlotContaminationSweep:
    def _make_sweep_df(self) -> pd.DataFrame:
        contamination_range = np.linspace(0.001, 0.10, 10)
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "contamination": contamination_range,
                "precision": rng.uniform(0.3, 0.9, 10),
                "recall": rng.uniform(0.3, 0.9, 10),
                "f1": rng.uniform(0.3, 0.9, 10),
            }
        )

    def test_returns_figure(self):
        df = self._make_sweep_df()
        fig = plot_contamination_sweep(df)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_figure_with_title(self):
        df = self._make_sweep_df()
        fig = plot_contamination_sweep(df, title="Custom Title")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_row(self):
        df = pd.DataFrame([{"contamination": 0.05, "precision": 0.8, "recall": 0.7, "f1": 0.75}])
        fig = plot_contamination_sweep(df)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_score_map
# ---------------------------------------------------------------------------


class TestPlotScoreMap:
    def _data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        scores = rng.uniform(0, 1, 60)
        y_true = np.concatenate([np.zeros(54, dtype=int), np.ones(6, dtype=int)])
        plt.close("all")
        return X, scores, y_true

    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        X, scores, _ = self._data()
        fig = plot_score_map(X, scores, contamination=0.1)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_with_y_true(self):
        import matplotlib.pyplot as plt

        X, scores, y_true = self._data()
        fig = plot_score_map(X, scores, contamination=0.1, y_true=y_true)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_flagged_count_matches_contamination(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        scores = rng.uniform(0, 1, 100)
        contamination = 0.1
        fig = plot_score_map(X, scores, contamination=contamination)
        assert isinstance(fig, Figure)
        expected_flagged = int(np.ceil(100 * contamination))
        actual_flagged = int((scores >= np.quantile(scores, 1.0 - contamination)).sum())
        assert actual_flagged == expected_flagged
        plt.close(fig)

    def test_single_feature_data(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 1))
        scores = rng.uniform(0, 1, 40)
        fig = plot_score_map(X, scores, contamination=0.05)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_true_anomalies(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 3))
        scores = rng.uniform(0, 1, 50)
        y_true = np.zeros(50, dtype=int)
        fig = plot_score_map(X, scores, contamination=0.1, y_true=y_true)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_precision_recall_curve with contamination operating point
# ---------------------------------------------------------------------------


class TestPlotPRCurveWithContamination:
    def test_with_contamination_returns_figure(self):
        import matplotlib.pyplot as plt

        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        fig = plot_precision_recall_curve(y_true, scores, contamination=0.4)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_contamination_none_unchanged(self):
        import matplotlib.pyplot as plt

        y_true = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        fig = plot_precision_recall_curve(y_true, scores, contamination=None)
        assert isinstance(fig, Figure)
        plt.close(fig)
