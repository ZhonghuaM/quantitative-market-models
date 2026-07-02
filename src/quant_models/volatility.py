"""Volatility models for market-risk research."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def garch11_variance(
    returns: pd.Series,
    omega: float,
    alpha: float,
    beta: float,
) -> pd.Series:
    """Generate conditional variance for a GARCH(1,1) model."""

    clean = returns.astype(float).dropna()
    variance = pd.Series(index=clean.index, dtype=float)
    unconditional = clean.var(ddof=0)
    variance.iloc[0] = unconditional
    for i in range(1, len(clean)):
        variance.iloc[i] = omega + alpha * clean.iloc[i - 1] ** 2 + beta * variance.iloc[i - 1]
    variance.name = "garch11_variance"
    return variance


def fit_garch11(returns: pd.Series) -> dict[str, float | pd.Series]:
    """Fit a Gaussian GARCH(1,1) model by maximum likelihood."""

    clean = returns.astype(float).dropna()
    if len(clean) < 50:
        raise ValueError("GARCH fitting needs at least 50 observations")

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
            return 1e12
        variance = garch11_variance(clean, omega, alpha, beta).clip(lower=1e-12)
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + clean**2 / variance))

    sample_var = float(clean.var(ddof=0))
    initial = np.array([sample_var * 0.05, 0.08, 0.90])
    bounds = [(1e-12, sample_var), (1e-6, 0.5), (1e-6, 0.999)]
    result = minimize(objective, initial, method="Nelder-Mead", bounds=bounds)
    if not result.success:
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = result.x
    variance = garch11_variance(clean, float(omega), float(alpha), float(beta))
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "log_likelihood": float(-objective(result.x)),
        "conditional_volatility": np.sqrt(variance) * np.sqrt(252.0),
    }


def volatility_regime_labels(
    volatility: pd.Series,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    """Label low, medium, and high volatility regimes."""

    clean = volatility.dropna()
    low = clean.quantile(low_quantile)
    high = clean.quantile(high_quantile)
    labels = pd.Series("medium", index=clean.index, name="volatility_regime")
    labels.loc[clean <= low] = "low"
    labels.loc[clean >= high] = "high"
    return labels
