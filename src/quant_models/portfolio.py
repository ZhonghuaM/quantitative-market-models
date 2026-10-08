"""Portfolio optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


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


def risk_contribution(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return each asset's fractional contribution to portfolio variance."""

    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    marginal = cov @ weights
    total_variance = float(weights @ marginal)
    if total_variance <= 0.0:
        raise ValueError("portfolio variance must be positive")
    return weights * marginal / total_variance


def risk_parity_portfolio(covariance: np.ndarray) -> np.ndarray:
    """Long-only equal-risk-contribution portfolio."""

    cov = np.asarray(covariance, dtype=float)
    n_assets = cov.shape[0]
    target = np.ones(n_assets) / n_assets

    def objective(weights: np.ndarray) -> float:
        contribution = risk_contribution(weights, cov)
        return float(np.sum((contribution - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    bounds = [(0.0, 1.0)] * n_assets
    initial = np.ones(n_assets) / n_assets
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"risk parity optimization failed: {result.message}")
    return result.x


def cvar_minimization_portfolio(
    return_scenarios: pd.DataFrame,
    confidence: float = 0.95,
    max_weight: float = 0.7,
) -> np.ndarray:
    """Long-only portfolio that minimizes historical expected shortfall."""

    scenarios = return_scenarios.dropna(how="any")
    n_assets = scenarios.shape[1]
    alpha = 1.0 - confidence

    def expected_shortfall_loss(weights: np.ndarray) -> float:
        portfolio_return = scenarios.to_numpy() @ weights
        losses = -portfolio_return
        cutoff = np.quantile(losses, confidence)
        tail = losses[losses >= cutoff]
        return float(tail.mean() if len(tail) else cutoff)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    bounds = [(0.0, max_weight)] * n_assets
    initial = np.ones(n_assets) / n_assets
    result = minimize(
        expected_shortfall_loss,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": alpha * 1e-8, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return result.x


def hierarchical_risk_parity_portfolio(covariance: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Compute HRP weights in the original covariance row/column order."""

    cov = pd.DataFrame(covariance).astype(float)
    if not cov.index.is_unique or not cov.index.equals(cov.columns):
        raise ValueError("covariance must have unique matching row and column labels")
    corr = cov.copy()
    std = np.sqrt(np.diag(cov))
    corr.iloc[:, :] = cov.to_numpy() / np.outer(std, std)
    distance = np.sqrt((1.0 - corr.clip(-1.0, 1.0)) / 2.0)
    condensed = squareform(distance.to_numpy(), checks=False)
    link = linkage(condensed, method="single")
    order = _quasi_diag(link)
    ordered_cov = cov.iloc[order, order]
    weights = pd.Series(1.0, index=ordered_cov.index)
    clusters = [ordered_cov.index.tolist()]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_var = _cluster_variance(ordered_cov, left)
        right_var = _cluster_variance(ordered_cov, right)
        allocation_left = 1.0 - left_var / (left_var + right_var)
        weights[left] *= allocation_left
        weights[right] *= 1.0 - allocation_left
        clusters.extend([left, right])
    result = weights.reindex(cov.index).to_numpy()
    return result / result.sum()


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by traversing a hierarchical linkage matrix."""

    link = link.astype(int)
    sort_index = pd.Series([link[-1, 0], link[-1, 1]])
    n_items = link[-1, 3]
    while sort_index.max() >= n_items:
        sort_index.index = range(0, sort_index.shape[0] * 2, 2)
        clusters = sort_index[sort_index >= n_items]
        i = clusters.index
        j = clusters.values - n_items
        sort_index[i] = link[j, 0]
        right = pd.Series(link[j, 1], index=i + 1)
        sort_index = pd.concat([sort_index, right]).sort_index()
        sort_index.index = range(sort_index.shape[0])
    return sort_index.astype(int).tolist()


def _cluster_variance(covariance: pd.DataFrame, cluster: list[int]) -> float:
    cluster_cov = covariance.loc[cluster, cluster].to_numpy()
    inverse_diag = 1.0 / np.diag(cluster_cov)
    weights = inverse_diag / inverse_diag.sum()
    return float(weights @ cluster_cov @ weights)


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
