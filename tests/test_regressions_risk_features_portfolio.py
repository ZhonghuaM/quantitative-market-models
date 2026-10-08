"""Regression cases for forecast timing, likelihood boundaries and label alignment."""

import numpy as np
import pandas as pd
import pytest

from quant_models.features import build_trend_dataset, relative_strength_index
from quant_models.portfolio import hierarchical_risk_parity_portfolio
from quant_models.risk import ewma_var_backtest, ewma_volatility, kupiec_pof_test


def test_ewma_calibration_is_excluded_and_forecast_precedes_observation():
    returns = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5])
    volatility = ewma_volatility(returns, lambda_=0.5, calibration_window=2)
    assert volatility.iloc[:2].isna().all()
    # Population variance of the two past calibration returns is 0.0225.
    assert volatility.iloc[2] == pytest.approx(0.15)
    assert volatility.iloc[3] ** 2 == pytest.approx(0.05625)

    changed = returns.copy()
    changed.iloc[2:] = [0.9, 0.8, 0.7]
    pd.testing.assert_series_equal(
        volatility.iloc[:3],
        ewma_volatility(changed, lambda_=0.5, calibration_window=2).iloc[:3],
    )


def test_ewma_future_edits_cannot_change_earlier_forecasts():
    returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100))
    changed = returns.copy()
    changed.iloc[-1] = 0.8
    # Even the final forecast precedes the edited final observation.
    pd.testing.assert_series_equal(ewma_volatility(returns), ewma_volatility(changed))
    assert ewma_volatility(returns).notna().sum() == 80


def test_ewma_accepts_external_initial_variance_and_short_inputs():
    volatility = ewma_volatility(pd.Series([0.1, -0.2]), lambda_=0.5, initial_variance=0.04)
    assert volatility.iloc[0] == pytest.approx(0.2)
    assert volatility.iloc[1] ** 2 == pytest.approx(0.025)
    assert ewma_volatility(pd.Series([0.01])).isna().all()
    assert ewma_volatility(pd.Series(dtype=float)).empty


@pytest.mark.parametrize("initial_variance", [-1.0, np.inf, np.nan])
def test_ewma_rejects_invalid_initial_variance(initial_variance):
    with pytest.raises(ValueError, match="initial_variance"):
        ewma_volatility(pd.Series([0.01]), initial_variance=initial_variance)


def test_var_backtest_uses_after_close_variance_and_exact_next_return():
    returns = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    frame = ewma_var_backtest(returns, lambda_=0.5, calibration_window=2)
    assert frame.index.tolist() == [2, 3, 4]
    assert frame.loc[2, "ewma_volatility"] ** 2 == pytest.approx(0.05625)
    np.testing.assert_allclose(frame["future_return"], [-0.4, 0.5, -0.6])
    np.testing.assert_allclose(frame["future_loss"], [0.4, -0.5, 0.6])

    changed = returns.copy()
    changed.iloc[3:] = [-0.9, 0.8, 0.7]
    other = ewma_var_backtest(changed, lambda_=0.5, calibration_window=2)
    assert other.loc[2, "var"] == frame.loc[2, "var"]
    assert other.loc[2, "future_return"] != frame.loc[2, "future_return"]


def test_var_backtest_multiperiod_targets_and_default_calibration():
    returns = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    frame = ewma_var_backtest(returns, horizon=2, calibration_window=2)
    assert frame.index.tolist() == [2, 3]
    np.testing.assert_allclose(frame["future_return"], [0.6 * 1.5 - 1, 1.5 * 0.4 - 1])
    default = ewma_var_backtest(pd.Series(np.linspace(-0.02, 0.03, 30)))
    assert default.index.tolist() == list(range(20, 29))


@pytest.mark.parametrize(
    "breach_count, expected_lr, expected_p",
    [
        (0, 20.100671707002885, 7.347086770069005e-6),
        (10, 0.0, 1.0),
        (20, 7.827239152922488, 0.00514646498249972),
        (1000, 9210.340371976183, 0.0),
    ],
)
def test_kupiec_boundary_and_interior_likelihoods(breach_count, expected_lr, expected_p):
    breaches = pd.Series([True] * breach_count + [False] * (1000 - breach_count))
    result = kupiec_pof_test(breaches, expected_probability=0.01)
    assert result["observations"] == 1000
    assert result["breaches"] == breach_count
    assert result["lr_statistic"] == pytest.approx(expected_lr, abs=1e-12)
    assert result["p_value"] == pytest.approx(expected_p, abs=1e-15)


def test_kupiec_excludes_missing_before_boolean_conversion():
    breaches = pd.Series([False, None, True, np.nan, pd.NA], dtype=object)
    result = kupiec_pof_test(breaches, expected_probability=0.5)
    assert result["observations"] == 2
    assert result["breaches"] == 1
    assert result["p_value"] == 1.0


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_kupiec_rejects_invalid_expected_probability(probability):
    with pytest.raises(ValueError, match="expected_probability"):
        kupiec_pof_test(pd.Series([True, False]), probability)


def test_kupiec_rejects_empty_or_nonbinary_indicators():
    with pytest.raises(ValueError, match="empty"):
        kupiec_pof_test(pd.Series([np.nan]), 0.01)
    with pytest.raises(ValueError, match="indicators"):
        kupiec_pof_test(pd.Series([0, 2]), 0.01)


@pytest.mark.parametrize("step, expected_rsi", [(1.0, 100.0), (-1.0, 0.0), (0.0, 50.0)])
def test_rsi_rising_falling_and_flat_series(step, expected_rsi):
    rsi = relative_strength_index(pd.Series(100.0 + step * np.arange(40)))
    assert rsi.iloc[:14].isna().all()
    assert len(rsi.dropna()) == 26
    np.testing.assert_allclose(rsi.dropna(), expected_rsi)


def _ohlcv_fixture():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.linspace(1000.0, 2000.0, len(dates)),
        }
    )


def test_trend_dataset_has_rows_and_exact_forward_alignment():
    ohlcv = _ohlcv_fixture()
    dataset, features = build_trend_dataset(ohlcv)
    assert features
    assert len(dataset) == 59
    assert dataset["target_up"].eq(1).all()
    expected = (ohlcv["close"].shift(-1) / ohlcv["close"] - 1.0).loc[dataset.index]
    np.testing.assert_allclose(dataset["forward_return"], expected, rtol=0, atol=1e-15)
    assert dataset.index[-1] == ohlcv.index[-2]


def test_trend_features_are_unchanged_by_future_price_edits():
    ohlcv = _ohlcv_fixture()
    changed = ohlcv.copy()
    changed.loc[changed.index[60]:, ["open", "high", "low", "close", "volume"]] *= 1.5
    original, features = build_trend_dataset(ohlcv)
    perturbed, _ = build_trend_dataset(changed)
    earlier = original.index[original.index < ohlcv.index[60]]
    assert len(earlier) == 40
    pd.testing.assert_frame_equal(original.loc[earlier, features], perturbed.loc[earlier, features])
    assert original.loc[ohlcv.index[59], "forward_return"] != perturbed.loc[
        ohlcv.index[59], "forward_return"
    ]


def test_hrp_preserves_unsorted_asset_labels_and_positional_alignment():
    variances = np.array([4.0, 1.0, 9.0])
    labels = ["Zulu", "Alpha", "Mike"]
    covariance = pd.DataFrame(np.diag(variances), index=labels, columns=labels)
    expected = (1.0 / variances) / (1.0 / variances).sum()
    np.testing.assert_allclose(hierarchical_risk_parity_portfolio(covariance), expected)
    np.testing.assert_allclose(hierarchical_risk_parity_portfolio(covariance.to_numpy()), expected)

    reordered = covariance.loc[labels[::-1], labels[::-1]]
    np.testing.assert_allclose(hierarchical_risk_parity_portfolio(reordered), expected[::-1])


def test_hrp_rejects_misaligned_covariance_labels():
    covariance = pd.DataFrame([[1.0, 0.1], [0.1, 2.0]], index=["b", "a"], columns=["a", "b"])
    with pytest.raises(ValueError, match="matching row and column labels"):
        hierarchical_risk_parity_portfolio(covariance)
