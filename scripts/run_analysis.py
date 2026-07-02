#!/usr/bin/env python
"""Run the full quantitative-model analysis pipeline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_models.backtest import backtest_probability_signal, performance_metrics
from quant_models.data import close_to_close_returns, load_ohlcv, load_price_series
from quant_models.features import build_trend_dataset
from quant_models.ml import walk_forward_classification
from quant_models.numerics import finite_difference_bvp, monte_carlo_integral
from quant_models.options import asian_arithmetic_call_mc, black_scholes_call, crr_binomial_call
from quant_models.portfolio import (
    default_asset_assumptions,
    global_minimum_variance_portfolio,
    portfolio_return,
    portfolio_volatility,
    random_long_only_portfolios,
    tangency_portfolio,
)
from quant_models.risk import ewma_var_backtest, kupiec_pof_test
from quant_models.stochastic import simulate_gbm_euler_milstein


REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_trading_model() -> dict[str, object]:
    ohlcv = load_ohlcv()
    dataset, features = build_trend_dataset(ohlcv)
    result = walk_forward_classification(
        dataset,
        features,
        train_size=504,
        test_size=63,
    )
    backtest = backtest_probability_signal(
        result.predictions,
        long_threshold=0.53,
        short_threshold=0.47,
        allow_short=False,
        transaction_cost_bps=5.0,
    )

    backtest.to_csv(REPORT_DIR / "signal_backtest.csv", index_label="date")
    result.feature_importance.to_csv(REPORT_DIR / "feature_importance.csv", header=["importance"])

    strategy_metrics = performance_metrics(backtest["strategy_return"])
    benchmark_metrics = performance_metrics(backtest["benchmark_return"])
    exposure = float((backtest["position"].abs() > 0.0).mean())
    trades = float(backtest["position"].diff().abs().fillna(backtest["position"].abs()).sum())

    plt.figure(figsize=(10, 5))
    plt.plot(backtest.index, backtest["strategy_equity"], label="ML signal")
    plt.plot(backtest.index, backtest["benchmark_equity"], label="Buy and hold")
    plt.title("Walk-forward trading equity curve")
    plt.ylabel("Growth of 1.00")
    plt.legend()
    _save_plot(FIGURE_DIR / "trading_equity_curve.png")

    plt.figure(figsize=(10, 4))
    plt.plot(backtest.index, backtest["strategy_drawdown"], label="ML signal")
    plt.plot(backtest.index, backtest["benchmark_drawdown"], label="Buy and hold")
    plt.title("Drawdown comparison")
    plt.ylabel("Drawdown")
    plt.legend()
    _save_plot(FIGURE_DIR / "trading_drawdown.png")

    plt.figure(figsize=(8, 5))
    result.feature_importance.sort_values().tail(10).plot(kind="barh")
    plt.title("Top model features")
    plt.xlabel("Mean random-forest importance")
    _save_plot(FIGURE_DIR / "feature_importance.png")

    return {
        "model": result.metrics,
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "exposure": exposure,
        "trades": trades,
        "rows": float(len(backtest)),
        "start": backtest.index.min(),
        "end": backtest.index.max(),
    }


def run_risk_model() -> dict[str, object]:
    sp500 = load_price_series()
    returns = close_to_close_returns(sp500)
    var_frame = ewma_var_backtest(returns, confidence=0.99, horizon=10, lambda_=0.94)
    var_frame.to_csv(REPORT_DIR / "ewma_var_backtest.csv", index_label="date")
    kupiec = kupiec_pof_test(var_frame["breach"], expected_probability=0.01)

    plt.figure(figsize=(10, 5))
    plt.plot(var_frame.index, var_frame["future_return"], label="Future 10-day return", lw=1.0)
    plt.plot(var_frame.index, -var_frame["var"], label="EWMA 99% VaR threshold", lw=1.0)
    breached = var_frame[var_frame["breach"]]
    plt.scatter(breached.index, breached["future_return"], s=12, color="red", label="Breach")
    plt.title("EWMA VaR backtest")
    plt.ylabel("Return")
    plt.legend()
    _save_plot(FIGURE_DIR / "ewma_var_backtest.png")

    return {
        "confidence": 0.99,
        "horizon_days": 10.0,
        "lambda": 0.94,
        "kupiec": kupiec,
    }


def run_portfolio_model() -> dict[str, object]:
    expected_returns, covariance, names = default_asset_assumptions()
    samples = random_long_only_portfolios(expected_returns, covariance, n_portfolios=6000)
    samples.to_csv(REPORT_DIR / "portfolio_samples.csv", index=False)

    tangency = tangency_portfolio(expected_returns, covariance, risk_free_rate=0.01)
    gmv = global_minimum_variance_portfolio(covariance)
    summary = pd.DataFrame(
        [
            {
                "portfolio": "tangency",
                "expected_return": portfolio_return(tangency, expected_returns),
                "volatility": portfolio_volatility(tangency, covariance),
                **{name: weight for name, weight in zip(names, tangency)},
            },
            {
                "portfolio": "global_minimum_variance",
                "expected_return": portfolio_return(gmv, expected_returns),
                "volatility": portfolio_volatility(gmv, covariance),
                **{name: weight for name, weight in zip(names, gmv)},
            },
        ]
    )
    summary.to_csv(REPORT_DIR / "portfolio_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        samples["volatility"],
        samples["expected_return"],
        c=samples["sharpe"],
        s=8,
        cmap="viridis",
        alpha=0.65,
    )
    plt.colorbar(scatter, label="Return / volatility")
    plt.scatter(
        [summary.loc[0, "volatility"]],
        [summary.loc[0, "expected_return"]],
        color="red",
        marker="*",
        s=180,
        label="Tangency",
    )
    plt.scatter(
        [summary.loc[1, "volatility"]],
        [summary.loc[1, "expected_return"]],
        color="black",
        marker="X",
        s=90,
        label="GMV",
    )
    plt.title("Sampled efficient frontier")
    plt.xlabel("Annual volatility")
    plt.ylabel("Annual expected return")
    plt.legend()
    _save_plot(FIGURE_DIR / "efficient_frontier.png")

    return {
        "tangency": summary.iloc[0].to_dict(),
        "global_minimum_variance": summary.iloc[1].to_dict(),
    }


def run_option_and_simulation_models() -> dict[str, object]:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.20
    maturity = 1.0
    bs_price = black_scholes_call(spot, strike, rate, volatility, maturity)
    steps = np.arange(4, 201, 4)
    binomial = np.array(
        [crr_binomial_call(spot, strike, rate, volatility, maturity, int(step)) for step in steps]
    )
    option_table = pd.DataFrame({"steps": steps, "binomial_call": binomial, "black_scholes": bs_price})
    option_table.to_csv(REPORT_DIR / "option_convergence.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, binomial, label="CRR binomial")
    plt.axhline(bs_price, color="black", linestyle="--", label="Black-Scholes")
    plt.title("European call convergence")
    plt.xlabel("Tree steps")
    plt.ylabel("Option value")
    plt.legend()
    _save_plot(FIGURE_DIR / "option_convergence.png")

    gbm = simulate_gbm_euler_milstein(spot, drift=rate, volatility=volatility, maturity=1.0, steps=252)
    gbm.to_csv(REPORT_DIR / "gbm_path_comparison.csv", index_label="time")

    plt.figure(figsize=(8, 5))
    plt.plot(gbm.index, gbm["exact"], label="Exact")
    plt.plot(gbm.index, gbm["euler"], label="Euler-Maruyama", alpha=0.8)
    plt.plot(gbm.index, gbm["milstein"], label="Milstein", alpha=0.8)
    plt.title("GBM path approximations")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    _save_plot(FIGURE_DIR / "gbm_path_comparison.png")

    asian = asian_arithmetic_call_mc(spot, strike, rate, volatility, maturity, steps=12, paths=30_000)
    return {
        "black_scholes_call": bs_price,
        "binomial_200_step": float(binomial[-1]),
        "asian_arithmetic_call_mc": asian,
    }


def run_numerical_examples() -> dict[str, object]:
    bvp = finite_difference_bvp(
        a=1.0,
        b=2.0,
        alpha=1.0,
        beta=6.0,
        n=50,
        p=lambda x: 3.0,
        q=lambda x: 2.0,
        f=lambda x: 4.0 * x**2,
    )
    bvp.to_csv(REPORT_DIR / "finite_difference_bvp.csv", index=False)

    mc = monte_carlo_integral(
        function=lambda x: x**2,
        lower=1.0,
        upper=3.0,
        samples=100_000,
    )
    mc["exact"] = 26.0 / 3.0
    mc["absolute_error"] = abs(mc["estimate"] - mc["exact"])
    return {"monte_carlo_integral_x2": mc}


def write_summary(metrics: dict[str, object]) -> None:
    trading = metrics["trading"]
    strategy = trading["strategy"]
    benchmark = trading["benchmark"]
    risk = metrics["risk"]["kupiec"]
    options = metrics["options_and_simulation"]

    lines = [
        "# Quantitative Model Analysis Summary",
        "",
        "This run regenerates the repository reports from the bundled sample data.",
        "",
        "## Walk-forward trading model",
        "",
        f"- Out-of-sample period: {trading['start'].date()} to {trading['end'].date()}",
        f"- Classification accuracy: {_pct(trading['model']['accuracy'])}",
        f"- ROC AUC: {trading['model'].get('roc_auc', float('nan')):.3f}",
        f"- Strategy total return: {_pct(strategy['total_return'])}",
        f"- Strategy annual volatility: {_pct(strategy['annual_volatility'])}",
        f"- Strategy Sharpe ratio: {strategy['sharpe']:.3f}",
        f"- Strategy max drawdown: {_pct(strategy['max_drawdown'])}",
        f"- Buy-and-hold total return over same rows: {_pct(benchmark['total_return'])}",
        f"- Average market exposure: {_pct(trading['exposure'])}",
        "",
        "## Risk model",
        "",
        f"- EWMA VaR observed breach rate: {_pct(risk['observed_breach_rate'])}",
        f"- Kupiec POF p-value: {risk['p_value']:.3f}",
        "",
        "## Option and simulation checks",
        "",
        f"- Black-Scholes call value: {options['black_scholes_call']:.4f}",
        f"- 200-step CRR value: {options['binomial_200_step']:.4f}",
        f"- Asian call Monte Carlo value: {options['asian_arithmetic_call_mc']['price']:.4f}",
        "",
        "Research code only. Results are historical and illustrative, not investment advice.",
        "",
    ]
    (REPORT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "trading": run_trading_model(),
        "risk": run_risk_model(),
        "portfolio": run_portfolio_model(),
        "options_and_simulation": run_option_and_simulation_models(),
        "numerics": run_numerical_examples(),
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(metrics)


if __name__ == "__main__":
    main()
