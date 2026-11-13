"""Independent NumPy/SciPy reference calculations; no native-library imports."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


def read_prices(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    return header[1:], [row[0] for row in rows], np.array([[float(x) for x in row[1:]] for row in rows])


def simple_returns(prices: np.ndarray) -> np.ndarray:
    return prices[1:] / prices[:-1] - 1.0


def historical_tail(returns: np.ndarray, confidence: float) -> tuple[float, float]:
    """Type-7 loss quantile and integral ES for the empirical distribution.

    ES integrates the empirical inverse CDF from confidence to 1. It includes
    fractional boundary mass; it is not the mean of observations above a
    linearly interpolated VaR threshold. Loss signs are preserved.
    """
    losses = -np.asarray(returns, dtype=float)
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError("returns must be a nonempty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    var = float(np.quantile(losses, confidence, method="linear"))
    # Integrate the overlap of each empirical probability bin with [c, 1].
    # This construction is independent of the native largest-k algorithm.
    ordered = np.sort(losses)
    right = np.arange(1, len(losses) + 1, dtype=float) / len(losses)
    left = np.arange(len(losses), dtype=float) / len(losses)
    tail_mass = np.maximum(right - np.maximum(left, confidence), 0.0)
    es = float(tail_mass @ ordered / (1.0 - confidence))
    return var, es


def risk_metrics(returns: np.ndarray, confidence: float) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    var, es = historical_tail(returns, confidence)
    total = float(np.prod(1.0 + returns) - 1.0)
    return {
        "observations": float(len(returns)),
        "mean_daily_return": float(returns.mean()),
        "volatility_daily": float(returns.std(ddof=1)),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252.0)),
        "total_return": total,
        "annualized_return": float((1.0 + total) ** (252.0 / len(returns)) - 1.0),
        "historical_var": var,
        "historical_es": es,
    }


def optimise_reference(returns: np.ndarray, target: float | None = None) -> tuple[np.ndarray, float]:
    """Long-only minimum variance with exactly 1e-8 added to daily covariance."""
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    mean = np.mean(returns, axis=0)
    count = len(mean)
    initial = np.full(count, 1.0 / count)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(count)}]
    if target is not None:
        lower, upper = int(np.argmin(mean)), int(np.argmax(mean))
        if np.isclose(target, mean[lower], rtol=0, atol=1e-14):
            initial = np.eye(count)[lower]
        elif np.isclose(target, mean[upper], rtol=0, atol=1e-14):
            initial = np.eye(count)[upper]
        else:
            share = (target - mean[lower]) / (mean[upper] - mean[lower])
            initial = np.zeros(count)
            initial[lower], initial[upper] = 1.0 - share, share
        # Scale the return equality to keep it numerically comparable to sum(w).
        constraints.append({"type": "eq", "fun": lambda w: (mean @ w - target) * 1e4, "jac": lambda w: mean * 1e4})
    scale = 1e6
    result = minimize(
        lambda w: float(w @ covariance @ w) * scale,
        initial,
        jac=lambda w: 2.0 * covariance @ w * scale,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(f"Independent SLSQP reference failed: {result.message}")
    return result.x, float(result.x @ covariance @ result.x)


def kupiec_reference(breaches: int, observations: int, expected_probability: float) -> dict[str, float]:
    if observations < 1 or not 0 <= breaches <= observations or not 0 < expected_probability < 1:
        raise ValueError("invalid binomial observations or probability")
    observed = breaches / observations
    expected_ll = breaches * np.log(expected_probability) + (observations - breaches) * np.log1p(-expected_probability)
    observed_ll = 0.0
    if breaches:
        observed_ll += breaches * np.log(observed)
    if breaches < observations:
        observed_ll += (observations - breaches) * np.log1p(-observed)
    lr = max(0.0, float(2.0 * (observed_ll - expected_ll)))
    return {"lr_statistic": lr, "p_value": float(chi2.sf(lr, 1))}


def lognormal_moments(log_mean: float, log_variance: float, horizon: int) -> tuple[float, float, float]:
    """Return mean, variance and fourth central moment of one-asset gross returns."""
    mean, variance = horizon * log_mean, horizon * log_variance
    expected_gross = np.exp(mean + variance / 2.0)
    return_variance = np.expm1(variance) * np.exp(2.0 * mean + variance)
    excess_kurtosis = np.exp(4 * variance) + 2 * np.exp(3 * variance) + 3 * np.exp(2 * variance) - 6
    fourth = (excess_kurtosis + 3) * return_variance**2
    return float(expected_gross - 1), float(return_variance), float(fourth)
