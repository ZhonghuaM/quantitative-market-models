"""Portfolio optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def covariance_from_vol_corr(volatilities: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """Build a covariance matrix from volatilities and a correlation matrix."""

    vol = np.asarray(volatilities, dtype=float)
    corr = np.asarray(correlation, dtype=float)
    if corr.shape != (len(vol), len(vol)):
        raise ValueError("correlation shape must match volatilities")
    return np.outer(vol, vol) * corr


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(np.asarray(weights) @ np.asarray(expected_returns))


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(weights @ np.asarray(covariance, dtype=float) @ weights))


def tangency_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Unconstrained maximum-Sharpe portfolio, normalized to sum to one."""

    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    excess = mu - risk_free_rate
    inv_cov = np.linalg.pinv(cov)
    raw = inv_cov @ excess
    denominator = raw.sum()
    if np.isclose(denominator, 0.0):
        raise ValueError("tangency portfolio is undefined for these inputs")
    return raw / denominator


def global_minimum_variance_portfolio(covariance: np.ndarray) -> np.ndarray:
    """Unconstrained global minimum-variance portfolio."""

    cov = np.asarray(covariance, dtype=float)
    ones = np.ones(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    raw = inv_cov @ ones
    return raw / (ones @ inv_cov @ ones)


def target_return_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
) -> np.ndarray:
    """Unconstrained minimum-variance portfolio for a target expected return."""

    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    ones = np.ones(len(mu))
    inv_cov = np.linalg.pinv(cov)
    a = ones @ inv_cov @ ones
    b = ones @ inv_cov @ mu
    c = mu @ inv_cov @ mu
    matrix = np.array([[a, b], [b, c]])
    rhs = np.array([1.0, target_return])
    lambda_1, lambda_2 = np.linalg.solve(matrix, rhs)
    return inv_cov @ (lambda_1 * ones + lambda_2 * mu)


def random_long_only_portfolios(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    n_portfolios: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample long-only portfolios from a Dirichlet distribution."""

    rng = np.random.default_rng(seed)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    weights = rng.dirichlet(np.ones(len(mu)), size=n_portfolios)
    returns = weights @ mu
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpe = returns / volatilities
    frame = pd.DataFrame(weights, columns=[f"asset_{i + 1}" for i in range(len(mu))])
    frame["expected_return"] = returns
    frame["volatility"] = volatilities
    frame["sharpe"] = sharpe
    return frame


def default_asset_assumptions() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return a small stylized asset universe used in the examples."""

    expected_returns = np.array([0.02, 0.07, 0.15, 0.20])
    volatilities = np.array([0.05, 0.12, 0.17, 0.25])
    correlation = np.array(
        [
            [1.0, 0.3, 0.3, 0.3],
            [0.3, 1.0, 0.6, 0.6],
            [0.3, 0.6, 1.0, 0.6],
            [0.3, 0.6, 0.6, 1.0],
        ]
    )
    names = ["Defensive", "Income", "Balanced", "Growth"]
    return expected_returns, covariance_from_vol_corr(volatilities, correlation), names
