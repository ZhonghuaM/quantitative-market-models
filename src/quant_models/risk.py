"""Market-risk diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    """Historical value-at-risk as a positive loss number."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    return float(-returns.dropna().quantile(1.0 - confidence))


def historical_expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    """Historical expected shortfall as a positive loss number."""

    clean = returns.dropna()
    var = historical_var(clean, confidence)
    tail_losses = -clean[clean <= -var]
    return float(tail_losses.mean())


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA volatility estimate."""

    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be between zero and one")
    clean = returns.astype(float).dropna()
    variance = pd.Series(index=clean.index, dtype=float)
    variance.iloc[0] = clean.var(ddof=0)
    for i in range(1, len(clean)):
        variance.iloc[i] = lambda_ * variance.iloc[i - 1] + (1.0 - lambda_) * clean.iloc[i - 1] ** 2
    volatility = np.sqrt(variance)
    volatility.name = "ewma_volatility"
    return volatility


def _future_compounded_return(returns: pd.Series, horizon: int) -> pd.Series:
    values = []
    clean = returns.astype(float)
    for i in range(len(clean)):
        future = clean.iloc[i + 1 : i + 1 + horizon]
        if len(future) < horizon:
            values.append(np.nan)
        else:
            values.append(float((1.0 + future).prod() - 1.0))
    return pd.Series(values, index=clean.index, name=f"future_{horizon}d_return")


def ewma_var_backtest(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 10,
    lambda_: float = 0.94,
) -> pd.DataFrame:
    """Backtest EWMA normal VaR against future horizon losses."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    clean = returns.astype(float).dropna()
    volatility = ewma_volatility(clean, lambda_=lambda_)
    var = norm.ppf(confidence) * volatility * math.sqrt(horizon)
    future_return = _future_compounded_return(clean, horizon)
    future_loss = -future_return
    frame = pd.DataFrame(
        {
            "return": clean,
            "ewma_volatility": volatility,
            "var": var,
            "future_return": future_return,
            "future_loss": future_loss,
        }
    ).dropna()
    frame["breach"] = frame["future_loss"] > frame["var"]
    return frame


def kupiec_pof_test(breaches: pd.Series, expected_probability: float) -> dict[str, float]:
    """Kupiec proportion-of-failures likelihood-ratio test."""

    clean = breaches.astype(bool).dropna()
    n = len(clean)
    x = int(clean.sum())
    if n == 0:
        raise ValueError("breaches cannot be empty")
    observed = x / n
    if x in (0, n):
        lr = 0.0
    else:
        log_likelihood_expected = (n - x) * np.log(1.0 - expected_probability) + x * np.log(
            expected_probability
        )
        log_likelihood_observed = (n - x) * np.log(1.0 - observed) + x * np.log(observed)
        lr = -2.0 * (log_likelihood_expected - log_likelihood_observed)
    return {
        "observations": float(n),
        "breaches": float(x),
        "expected_breach_rate": float(expected_probability),
        "observed_breach_rate": float(observed),
        "lr_statistic": float(lr),
        "p_value": float(1.0 - chi2.cdf(lr, df=1)),
    }
