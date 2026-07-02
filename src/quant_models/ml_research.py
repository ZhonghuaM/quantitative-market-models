"""Baseline-first ML research helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quant_models.ml import walk_forward_classification


def baseline_model_zoo(random_state: int = 42) -> dict[str, object]:
    """Return a compact but diverse classifier set for time-series model comparison."""

    return {
        "naive_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compare_walk_forward_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    train_size: int = 504,
    test_size: int = 63,
) -> pd.DataFrame:
    """Run walk-forward comparison across simple and nonlinear baselines."""

    rows = []
    for name, model in baseline_model_zoo().items():
        result = walk_forward_classification(
            dataset=dataset,
            feature_columns=feature_columns,
            train_size=train_size,
            test_size=test_size,
            model=clone(model),
        )
        predictions = result.predictions
        row = {
            "model": name,
            "accuracy": accuracy_score(predictions["actual_up"], predictions["predicted_up"]),
            "balanced_accuracy": balanced_accuracy_score(
                predictions["actual_up"], predictions["predicted_up"]
            ),
            "brier_score": brier_score_loss(
                predictions["actual_up"],
                predictions["probability_up"].clip(0.0, 1.0),
            ),
        }
        if predictions["actual_up"].nunique() == 2:
            row["roc_auc"] = roc_auc_score(predictions["actual_up"], predictions["probability_up"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def calibration_table(
    predictions: pd.DataFrame,
    probability_column: str = "probability_up",
    target_column: str = "actual_up",
    bins: int = 10,
) -> pd.DataFrame:
    """Summarise predicted probability calibration by equal-width bins."""

    frame = predictions[[probability_column, target_column]].copy()
    frame["bin"] = pd.cut(frame[probability_column], bins=np.linspace(0.0, 1.0, bins + 1))
    grouped = frame.groupby("bin", observed=False)
    table = grouped.agg(
        mean_probability=(probability_column, "mean"),
        observed_frequency=(target_column, "mean"),
        count=(target_column, "size"),
    ).dropna()
    return table.reset_index(drop=True)
