"""Machine-learning helpers for walk-forward market prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class WalkForwardResult:
    """Container returned by the walk-forward classifier."""

    predictions: pd.DataFrame
    feature_importance: pd.Series
    metrics: dict[str, float]


def make_default_classifier(random_state: int = 42) -> RandomForestClassifier:
    """Create a conservative tree ensemble for noisy daily direction data."""

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        # Fixed accumulation order keeps stored probabilities stable on reruns.
        n_jobs=1,
    )


def _probability_of_positive_class(model: BaseEstimator, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            return np.zeros(len(x_test))
        positive_index = classes.index(1)
        return probabilities[:, positive_index]

    decision = model.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def walk_forward_classification(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_up",
    train_size: int = 504,
    test_size: int = 63,
    step_size: int | None = None,
    model: BaseEstimator | None = None,
) -> WalkForwardResult:
    """Run expanding-window walk-forward classification."""

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if len(dataset) <= train_size:
        raise ValueError("dataset is too short for the requested train_size")

    step = test_size if step_size is None else step_size
    if step < test_size:
        raise ValueError("step_size must be at least test_size to avoid overlapping predictions")
    if not dataset.index.is_unique or not dataset.index.is_monotonic_increasing:
        raise ValueError("dataset must have a unique chronological index")
    estimator = model if model is not None else make_default_classifier()
    records: list[pd.DataFrame] = []
    importances: list[pd.Series] = []

    start = train_size
    fold = 0
    while start < len(dataset):
        end = min(start + test_size, len(dataset))
        train = dataset.iloc[:start]
        test = dataset.iloc[start:end]
        if test.empty:
            break

        fitted = clone(estimator)
        x_train = train[feature_columns]
        y_train = train[target_column]
        x_test = test[feature_columns]
        y_test = test[target_column]

        fitted.fit(x_train, y_train)
        probability_up = _probability_of_positive_class(fitted, x_test)
        prediction = (probability_up >= 0.5).astype(int)

        fold_records = pd.DataFrame(
            {
                "fold": fold,
                "actual_up": y_test.astype(int).values,
                "predicted_up": prediction,
                "probability_up": probability_up,
                "forward_return": test["forward_return"].values,
            },
            index=test.index,
        )
        records.append(fold_records)

        if hasattr(fitted, "feature_importances_"):
            importances.append(pd.Series(fitted.feature_importances_, index=feature_columns))

        fold += 1
        start += step

    predictions = pd.concat(records).sort_index()
    if importances:
        feature_importance = (
            pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        )
    else:
        feature_importance = pd.Series(dtype=float)

    metrics = {
        "accuracy": float(accuracy_score(predictions["actual_up"], predictions["predicted_up"])),
        "balanced_accuracy": float(
            balanced_accuracy_score(predictions["actual_up"], predictions["predicted_up"])
        ),
    }
    if predictions["actual_up"].nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(predictions["actual_up"], predictions["probability_up"])
        )

    return WalkForwardResult(
        predictions=predictions, feature_importance=feature_importance, metrics=metrics
    )
