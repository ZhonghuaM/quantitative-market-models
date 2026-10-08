import numpy as np
import pandas as pd

from quant_models.backtest import backtest_probability_signal, performance_metrics
from quant_models.factor_models import capm_regression, pca_statistical_factors
from quant_models.features import build_trend_dataset
from quant_models.macro import align_macro_to_market, fred_graph_csv_url
from quant_models.nlp_retrieval import TfidfRetriever, chunk_text, source_grounded_brief
from quant_models.options import (
    black_scholes_call,
    black_scholes_greeks,
    crr_binomial_call,
    implied_volatility_call,
)
from quant_models.portfolio import (
    cvar_minimization_portfolio,
    default_asset_assumptions,
    global_minimum_variance_portfolio,
    hierarchical_risk_parity_portfolio,
    portfolio_volatility,
    risk_parity_portfolio,
    tangency_portfolio,
)
from quant_models.risk import ewma_var_backtest, kupiec_pof_test, parametric_var_es
from quant_models.time_series import (
    ar1_fit,
    exponential_smoothing,
    kalman_local_level,
    pairs_spread_signal,
    regime_transition_matrix,
)
from quant_models.volatility import fit_garch11


def test_crr_binomial_converges_to_black_scholes_call():
    bs = black_scholes_call(100, 100, 0.05, 0.2, 1)
    crr = crr_binomial_call(100, 100, 0.05, 0.2, 1, 500)
    assert abs(bs - crr) < 0.01


def test_implied_volatility_and_greeks_are_consistent():
    price = black_scholes_call(100, 100, 0.05, 0.2, 1)
    implied = implied_volatility_call(price, 100, 100, 0.05, 1)
    greeks = black_scholes_greeks(100, 100, 0.05, 0.2, 1)
    assert abs(implied - 0.2) < 1e-5
    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["gamma"] > 0.0


def test_portfolio_weights_sum_to_one():
    expected_returns, covariance, _ = default_asset_assumptions()
    gmv = global_minimum_variance_portfolio(covariance)
    tangency = tangency_portfolio(expected_returns, covariance, risk_free_rate=0.01)
    risk_parity = risk_parity_portfolio(covariance)
    hrp = hierarchical_risk_parity_portfolio(covariance)
    assert np.isclose(gmv.sum(), 1.0)
    assert np.isclose(tangency.sum(), 1.0)
    assert np.isclose(risk_parity.sum(), 1.0)
    assert np.isclose(hrp.sum(), 1.0)
    assert portfolio_volatility(gmv, covariance) > 0.0


def test_cvar_minimization_returns_long_only_weights():
    rng = np.random.default_rng(42)
    scenarios = pd.DataFrame(rng.normal(0.001, 0.02, size=(300, 4)))
    weights = cvar_minimization_portfolio(scenarios, confidence=0.95)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= -1e-8).all()


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
    assert not dataset.empty
    pd.testing.assert_series_equal(
        dataset["forward_return"], close.pct_change().shift(-1).loc[dataset.index],
        check_names=False,
    )
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
    parametric = parametric_var_es(returns, horizon=5)
    assert parametric["var"] > 0.0
    assert parametric["expected_shortfall"] > parametric["var"]


def test_factor_and_volatility_helpers():
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=250)
    benchmark = pd.Series(rng.normal(0, 0.01, 250), index=index)
    asset = 0.0002 + 1.4 * benchmark + pd.Series(rng.normal(0, 0.005, 250), index=index)
    capm = capm_regression(asset, benchmark)
    assert 1.0 < capm["beta"] < 1.8

    returns = pd.DataFrame(
        {
            "a": benchmark,
            "b": asset,
            "c": pd.Series(rng.normal(0, 0.012, 250), index=index),
        }
    )
    _, loadings = pca_statistical_factors(returns, n_components=2)
    assert "explained_variance_ratio" in loadings.index

    garch = fit_garch11(benchmark)
    assert 0.0 <= garch["persistence"] < 0.999


def test_time_series_helpers():
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=180)
    base = pd.Series(np.cumsum(rng.normal(0, 1, len(index))), index=index)
    hedge = base + pd.Series(rng.normal(0, 0.3, len(index)), index=index)
    returns = base.diff().dropna()
    ar1 = ar1_fit(returns)
    smooth = exponential_smoothing(returns, alpha=0.2)
    kalman = kalman_local_level(returns)
    pair = pairs_spread_signal(base, hedge, window=30)
    regimes = regime_transition_matrix(pd.Series(["low", "low", "high", "low"]))
    assert "phi" in ar1
    assert len(smooth) == len(returns)
    assert "filtered_level" in kalman
    assert not pair.empty
    assert not regimes.empty


def test_retrieval_and_macro_helpers():
    text = "Liquidity risk affects credit access. Revenue growth came from renewals and demand."
    chunks = chunk_text(text, chunk_words=6, overlap_words=2)
    results = TfidfRetriever().fit(chunks).query("What affects liquidity risk?", top_k=1)
    brief = source_grounded_brief("What affects liquidity risk?", results)
    assert results[0].score > 0.0
    assert "Liquidity" in brief or "liquidity" in brief
    assert fred_graph_csv_url("DGS10").endswith("DGS10")

    market = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, freq="D"))
    macro = pd.Series([1.0, 2.0], index=pd.to_datetime(["2023-12-31", "2024-01-02"]), name="macro")
    aligned = align_macro_to_market(market, macro)
    assert aligned["macro"].tolist() == [1.0, 2.0, 2.0]
