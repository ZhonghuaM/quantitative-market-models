"""Baseline-first ML research helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quant_models.ml import WalkForwardResult, make_default_classifier, walk_forward_classification


def baseline_model_zoo(random_state: int = 42) -> dict[str, object]:
    """Return a compact but diverse classifier set for time-series model comparison."""

    return {
        "naive_most_frequent": DummyClassifier(strategy="most_frequent"),
        "training_frequency": DummyClassifier(strategy="prior"),
        "logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "random_forest": make_default_classifier(random_state=random_state),
    }


def evaluate_walk_forward_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    train_size: int = 504,
    test_size: int = 63,
    random_state: int = 42,
) -> dict[str, WalkForwardResult]:
    """Fit each model once; reuse these exact predictions for every diagnostic."""
    return {
        name: walk_forward_classification(
            dataset,
            feature_columns,
            train_size=train_size,
            test_size=test_size,
            model=clone(model),
        )
        for name, model in baseline_model_zoo(random_state).items()
    }


def summarise_model_predictions(results: dict[str, WalkForwardResult]) -> pd.DataFrame:
    """Summarise stored predictions without refitting or selecting a trading model."""
    rows = []
    for name, result in results.items():
        predictions = result.predictions
        row = {
            "model": name,
            "accuracy": accuracy_score(predictions["actual_up"], predictions["predicted_up"]),
            "balanced_accuracy": balanced_accuracy_score(
                predictions["actual_up"], predictions["predicted_up"]
            ),
            "brier_score": brier_score_loss(
                predictions["actual_up"],
                predictions["probability_up"],
            ),
            "observations": len(predictions),
        }
        if predictions["actual_up"].nunique() == 2:
            row["roc_auc"] = roc_auc_score(predictions["actual_up"], predictions["probability_up"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def compare_walk_forward_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    train_size: int = 504,
    test_size: int = 63,
) -> pd.DataFrame:
    """Convenience interface; use evaluate_walk_forward_models to keep predictions."""
    return summarise_model_predictions(
        evaluate_walk_forward_models(dataset, feature_columns, train_size, test_size)
    )


def paired_brier_block_bootstrap(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    block_size: int = 10,
    repetitions: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Paired moving-block interval for baseline loss minus model loss.

    Resamples contiguous blocks of the observed paired loss differences. This
    exploratory interval is conditional on the fitted models and sample; it
    neither refits models nor accounts for model/threshold selection. Dependence
    beyond the block length and nonstationarity can invalidate coverage.
    """
    if not predictions.index.equals(baseline.index):
        raise ValueError("model and baseline must have identical chronological rows")
    if not predictions.index.is_unique or not predictions.index.is_monotonic_increasing:
        raise ValueError("predictions must have a unique chronological index")
    if not predictions["actual_up"].equals(baseline["actual_up"]):
        raise ValueError("model and baseline targets must match")
    n = len(predictions)
    if not 1 <= block_size <= n or repetitions < 2:
        raise ValueError("require 1 <= block_size <= observations and repetitions >= 2")
    actual = predictions["actual_up"].to_numpy()
    delta = (baseline["probability_up"].to_numpy() - actual) ** 2 - (
        predictions["probability_up"].to_numpy() - actual
    ) ** 2
    if not np.isfinite(delta).all():
        raise ValueError("loss differences must be finite")
    rng = np.random.default_rng(random_state)
    starts = rng.integers(0, n - block_size + 1, size=(repetitions, int(np.ceil(n / block_size))))
    indices = (starts[..., None] + np.arange(block_size)).reshape(repetitions, -1)[:, :n]
    boot = delta[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "block_size": block_size,
        "repetitions": repetitions,
        "observations": n,
        "mean_brier_improvement": float(delta.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def calibration_table(
    predictions: pd.DataFrame,
    probability_column: str = "probability_up",
    target_column: str = "actual_up",
    bins: int = 10,
) -> pd.DataFrame:
    """Summarise predicted probability calibration by equal-width bins."""

    frame = predictions[[probability_column, target_column]].copy()
    frame["bin"] = pd.cut(
        frame[probability_column], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True
    )
    grouped = frame.groupby("bin", observed=False)
    table = grouped.agg(
        mean_probability=(probability_column, "mean"),
        observed_frequency=(target_column, "mean"),
        count=(target_column, "size"),
    ).dropna()
    return table.reset_index(drop=True)
