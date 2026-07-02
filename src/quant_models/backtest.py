"""Signal backtesting and performance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_probabilities(
    probability_up: pd.Series,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
) -> pd.Series:
    """Convert positive-class probabilities into long/flat or long/short positions."""

    if not 0.0 <= short_threshold <= long_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= short <= long <= 1")

    positions = pd.Series(0.0, index=probability_up.index, name="position")
    positions.loc[probability_up >= long_threshold] = 1.0
    if allow_short:
        positions.loc[probability_up <= short_threshold] = -1.0
    return positions


def equity_curve(returns: pd.Series) -> pd.Series:
    """Convert periodic returns to a growth-of-one equity curve."""

    curve = (1.0 + returns.fillna(0.0)).cumprod()
    curve.name = "equity"
    return curve


def drawdown(equity: pd.Series) -> pd.Series:
    """Compute percentage drawdown from running equity peaks."""

    running_peak = equity.cummax()
    dd = equity / running_peak - 1.0
    dd.name = "drawdown"
    return dd


def performance_metrics(
    returns: pd.Series,
    annual_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute common return, risk, and drawdown metrics."""

    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns cannot be empty")

    total_return = float((1.0 + clean).prod() - 1.0)
    years = len(clean) / annual_factor
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(annual_factor))
    excess_return = annual_return - risk_free_rate
    sharpe = float(excess_return / annual_volatility) if annual_volatility > 0 else np.nan
    curve = equity_curve(clean)
    max_drawdown = float(drawdown(curve).min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan
    win_rate = float((clean > 0).mean())

    return {
        "observations": float(len(clean)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
    }


def backtest_probability_signal(
    predictions: pd.DataFrame,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest a probability-based signal against next-period returns."""

    required = {"probability_up", "forward_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    result = predictions.copy()
    result["position"] = positions_from_probabilities(
        result["probability_up"],
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        allow_short=allow_short,
    )
    turnover = result["position"].diff().abs().fillna(result["position"].abs())
    cost = turnover * transaction_cost_bps / 10_000.0
    result["strategy_return_gross"] = result["position"] * result["forward_return"]
    result["transaction_cost"] = cost
    result["strategy_return"] = result["strategy_return_gross"] - cost
    result["benchmark_return"] = result["forward_return"]
    result["strategy_equity"] = equity_curve(result["strategy_return"])
    result["benchmark_equity"] = equity_curve(result["benchmark_return"])
    result["strategy_drawdown"] = drawdown(result["strategy_equity"])
    result["benchmark_drawdown"] = drawdown(result["benchmark_equity"])
    return result
