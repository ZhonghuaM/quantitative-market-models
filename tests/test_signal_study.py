"""Small, hand-checkable cases for timing and paired prediction diagnostics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_models.backtest import backtest_probability_signal, performance_metrics
from quant_models.features import build_trend_dataset
from quant_models.ml import walk_forward_classification
from quant_models.ml_research import calibration_table, paired_brier_block_bootstrap


def _signal_predictions():
    return pd.DataFrame(
        {
            "probability_up": [0.6, 0.4, 0.7, 0.8, 0.2],
            "forward_return": [0.01, -0.02, 0.015, -0.01, 0.005],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="B"),
    )


def test_default_delayed_execution_costs_entries_and_exits_once():
    result = backtest_probability_signal(_signal_predictions(), transaction_cost_bps=10)
    np.testing.assert_array_equal(result["signal_position"], [1, 0, 1, 1, 0])
    np.testing.assert_array_equal(result["position"], [0, 1, 0, 1, 1])
    np.testing.assert_allclose(result["transaction_cost"], [0, 0.001, 0.001, 0.001, 0])
    np.testing.assert_allclose(result["strategy_return"], [0, -0.021, -0.001, -0.011, 0.005])
    # The last holding remains open; the return does not include an extra liquidation fee.
    assert result["position"].iloc[-1] == 1
    assert result["transaction_cost"].iloc[-1] == 0


def test_same_close_diagnostic_reproduces_undelayed_signals():
    result = backtest_probability_signal(
        _signal_predictions(), transaction_cost_bps=10, execution_delay=0
    )
    np.testing.assert_array_equal(result["position"], [1, 0, 1, 1, 0])
    np.testing.assert_allclose(result["strategy_return"], [0.009, -0.001, 0.014, -0.01, -0.001])
    # Stopping while long also leaves the position open in the diagnostic clock.
    truncated = backtest_probability_signal(
        _signal_predictions().iloc[:4], transaction_cost_bps=10, execution_delay=0
    )
    assert truncated["position"].iloc[-1] == 1
    assert truncated["transaction_cost"].iloc[-1] == 0


def test_initial_loss_is_included_in_maximum_drawdown():
    metrics = performance_metrics(pd.Series([-0.1, 0.0, 0.0]))
    assert metrics["total_return"] == pytest.approx(-0.1)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)


def test_sharpe_uses_arithmetic_excess_return_with_compounded_risk_free_rate():
    # Four observations comprise one year. Their mean is zero and variance is 0.025.
    metrics = performance_metrics(
        pd.Series([0.1, -0.1, 0.2, -0.2]),
        annual_factor=4,
        risk_free_rate=1.01**4 - 1.0,
    )
    assert metrics["annual_volatility"] == pytest.approx(np.sqrt(0.1))
    assert metrics["sharpe"] == pytest.approx(-0.04 / np.sqrt(0.1))
    assert metrics["annual_return"] == pytest.approx(-0.0496)


def _classification_dataset():
    return pd.DataFrame(
        {
            "feature": np.arange(12, dtype=float),
            "target_up": [0, 1] * 6,
            "forward_return": [-0.01, 0.01] * 6,
        },
        index=pd.date_range("2024-01-01", periods=12, freq="B"),
    )


@pytest.mark.parametrize("step_size", [2, 0, -1])
def test_walk_forward_rejects_overlapping_or_nonpositive_steps(step_size):
    with pytest.raises(ValueError, match="step_size"):
        walk_forward_classification(
            _classification_dataset(), ["feature"], train_size=6, test_size=3,
            step_size=step_size, model=DummyClassifier(strategy="prior"),
        )


def test_walk_forward_default_folds_cover_each_future_date_once():
    dataset = _classification_dataset()
    result = walk_forward_classification(
        dataset, ["feature"], train_size=6, test_size=4,
        model=DummyClassifier(strategy="prior"),
    )
    pd.testing.assert_index_equal(result.predictions.index, dataset.index[6:])
    assert result.predictions.index.is_unique
    assert result.predictions["fold"].tolist() == [0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize("only_training_class", [0, 1])
def test_single_class_training_folds_report_positive_class_probability(only_training_class):
    dataset = _classification_dataset().iloc[:10].copy()
    dataset.iloc[:6, dataset.columns.get_loc("target_up")] = only_training_class
    result = walk_forward_classification(
        dataset, ["feature"], train_size=6, test_size=4,
        model=DummyClassifier(strategy="prior"),
    )
    assert len(result.predictions) == 4
    np.testing.assert_array_equal(result.predictions["probability_up"], only_training_class)
    np.testing.assert_array_equal(result.predictions["predicted_up"], only_training_class)


def test_calibration_counts_zero_and_one_probabilities():
    predictions = pd.DataFrame(
        {"probability_up": [0.0, 0.1, 0.9, 1.0], "actual_up": [0, 1, 0, 1]}
    )
    table = calibration_table(predictions, bins=2)
    assert table["count"].tolist() == [2, 2]
    assert table["count"].sum() == len(predictions)
    np.testing.assert_allclose(table["mean_probability"], [0.05, 0.95])
    np.testing.assert_allclose(table["observed_frequency"], [0.5, 0.5])


def _paired_predictions():
    predictions = pd.DataFrame(
        {"actual_up": [0, 1] * 6, "probability_up": [0.25, 0.75] * 6},
        index=pd.date_range("2024-01-01", periods=12, freq="B"),
    )
    baseline = predictions.assign(probability_up=0.5)
    return predictions, baseline


def test_identical_predictions_have_exactly_zero_paired_interval():
    predictions, _ = _paired_predictions()
    interval = paired_brier_block_bootstrap(
        predictions, predictions.copy(), block_size=3, repetitions=100
    )
    assert interval["mean_brier_improvement"] == 0
    assert interval["ci_95_lower"] == 0
    assert interval["ci_95_upper"] == 0
    assert interval["observations"] == 12


@pytest.mark.parametrize("block_size", [1, 3, 12])
def test_constant_paired_brier_improvement_is_exact(block_size):
    predictions, baseline = _paired_predictions()
    interval = paired_brier_block_bootstrap(
        predictions, baseline, block_size=block_size, repetitions=100
    )
    # Baseline loss is 1/4; model loss is 1/16 for every row.
    assert interval["mean_brier_improvement"] == 3 / 16
    assert interval["ci_95_lower"] == 3 / 16
    assert interval["ci_95_upper"] == 3 / 16


def test_full_length_block_preserves_the_observed_sequence():
    predictions, baseline = _paired_predictions()
    predictions["probability_up"] = predictions["actual_up"]
    baseline["probability_up"] = [1, 0] * 3 + [0, 1] * 3
    interval = paired_brier_block_bootstrap(
        predictions, baseline, block_size=12, repetitions=100
    )
    assert interval["mean_brier_improvement"] == 0.5
    assert interval["ci_95_lower"] == 0.5
    assert interval["ci_95_upper"] == 0.5


@pytest.mark.parametrize("mismatch", ["dates", "targets", "reverse_order", "duplicates"])
def test_paired_bootstrap_rejects_incompatible_or_nonchronological_rows(mismatch):
    predictions, baseline = _paired_predictions()
    if mismatch == "dates":
        baseline.index = baseline.index + pd.Timedelta(days=1)
    elif mismatch == "targets":
        baseline.iloc[0, baseline.columns.get_loc("actual_up")] = 1
    elif mismatch == "reverse_order":
        predictions = predictions.iloc[::-1]
        baseline = baseline.iloc[::-1]
    else:
        duplicate_dates = predictions.index.to_list()
        duplicate_dates[1] = duplicate_dates[0]
        predictions.index = duplicate_dates
        baseline.index = duplicate_dates
    with pytest.raises(ValueError):
        paired_brier_block_bootstrap(predictions, baseline, block_size=3, repetitions=100)


def test_future_market_edits_cannot_change_first_walk_forward_prediction():
    rng = np.random.default_rng(29)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 100)), index=dates)
    market = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(1000, 2000, size=100).astype(float),
        }
    )
    dataset, features = build_trend_dataset(market)
    first_prediction_date = dataset.index[40]
    perturbed_market = market.astype(float).copy()
    factor = 0.5 if dataset.loc[first_prediction_date, "target_up"] == 1 else 2.0
    perturbed_market.loc[perturbed_market.index > first_prediction_date] *= factor
    perturbed, _ = build_trend_dataset(perturbed_market)

    # The final training label describes t-1 -> t, so it is known at close t.
    pd.testing.assert_frame_equal(dataset.iloc[:40], perturbed.iloc[:40])
    assert dataset.loc[first_prediction_date, "target_up"] != perturbed.loc[
        first_prediction_date, "target_up"
    ]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    original = walk_forward_classification(
        dataset, features, train_size=40, test_size=10, model=model
    )
    changed = walk_forward_classification(
        perturbed, features, train_size=40, test_size=10, model=model
    )
    assert original.predictions.index[0] == first_prediction_date
    assert changed.predictions.index[0] == first_prediction_date
    assert original.predictions["probability_up"].iloc[0] == pytest.approx(
        changed.predictions["probability_up"].iloc[0], abs=1e-14
    )
