import numpy as np
import pandas as pd

from quant_models.backtest import backtest_probability_signal, performance_metrics
from quant_models.features import build_trend_dataset
from quant_models.options import black_scholes_call, crr_binomial_call
from quant_models.portfolio import (
    default_asset_assumptions,
    global_minimum_variance_portfolio,
    portfolio_volatility,
    tangency_portfolio,
)
from quant_models.risk import ewma_var_backtest, kupiec_pof_test


def test_crr_binomial_converges_to_black_scholes_call():
    bs = black_scholes_call(100, 100, 0.05, 0.2, 1)
    crr = crr_binomial_call(100, 100, 0.05, 0.2, 1, 500)
    assert abs(bs - crr) < 0.01


def test_portfolio_weights_sum_to_one():
    expected_returns, covariance, _ = default_asset_assumptions()
    gmv = global_minimum_variance_portfolio(covariance)
    tangency = tangency_portfolio(expected_returns, covariance, risk_free_rate=0.01)
    assert np.isclose(gmv.sum(), 1.0)
    assert np.isclose(tangency.sum(), 1.0)
    assert portfolio_volatility(gmv, covariance) > 0.0


def test_build_trend_dataset_uses_forward_return():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.linspace(1000, 2000, len(dates)),
        },
        index=dates,
    )
    dataset, features = build_trend_dataset(frame)
    assert features
    assert "forward_return" in dataset
    assert dataset["target_up"].eq(1).all()


def test_backtest_probability_signal_and_metrics():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    predictions = pd.DataFrame(
        {
            "probability_up": [0.6, 0.4, 0.7, 0.8, 0.2],
            "forward_return": [0.01, -0.02, 0.015, -0.01, 0.005],
        },
        index=index,
    )
    result = backtest_probability_signal(predictions, transaction_cost_bps=0)
    metrics = performance_metrics(result["strategy_return"])
    assert "strategy_equity" in result
    assert metrics["observations"] == 5.0


def test_ewma_var_backtest_and_kupiec():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 300), index=pd.date_range("2020-01-01", periods=300))
    var_frame = ewma_var_backtest(returns, horizon=5)
    test = kupiec_pof_test(var_frame["breach"], expected_probability=0.01)
    assert len(var_frame) > 0
    assert 0.0 <= test["observed_breach_rate"] <= 1.0
