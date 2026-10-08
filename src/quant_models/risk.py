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


def parametric_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
) -> dict[str, float]:
    """Normal parametric VaR and Expected Shortfall as positive loss numbers."""

    clean = returns.dropna()
    mu = float(clean.mean() * horizon)
    sigma = float(clean.std(ddof=1) * math.sqrt(horizon))
    z = norm.ppf(confidence)
    var = -(mu - z * sigma)
    es = -(mu - sigma * norm.pdf(z) / (1.0 - confidence))
    return {"var": float(var), "expected_shortfall": float(es)}


def rolling_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> pd.Series:
    """Rolling historical expected shortfall."""

    def _tail(series: pd.Series) -> float:
        var = -series.quantile(1.0 - confidence)
        losses = -series[series <= -var]
        return float(losses.mean()) if len(losses) else float(var)

    es = returns.dropna().rolling(window).apply(_tail, raw=False)
    es.name = f"rolling_es_{int(confidence * 100)}"
    return es.dropna()


def stress_scenario_table(
    returns: pd.Series,
    scenarios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply simple shock scenarios to a representative one-day return distribution."""

    clean = returns.dropna()
    base = {
        "median_day": float(clean.median()),
        "one_sigma_down": float(clean.mean() - clean.std(ddof=1)),
        "historical_5pct": float(clean.quantile(0.05)),
        "historical_1pct": float(clean.quantile(0.01)),
    }
    if scenarios:
        base.update(scenarios)
    frame = pd.DataFrame(
        [{"scenario": name, "return_shock": shock, "loss": -shock} for name, shock in base.items()]
    )
    return frame.sort_values("loss", ascending=False).reset_index(drop=True)


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    *,
    calibration_window: int = 20,
    initial_variance: float | None = None,
) -> pd.Series:
    """Forecast volatility at t using only returns strictly before t.

    By default, the first ``calibration_window`` nonmissing observations estimate
    the initial population variance; their forecasts are NaN and they are excluded
    from evaluation. The first forecast is for the following observation. An
    externally supplied ``initial_variance`` instead supplies the forecast for the
    first observation and must have been estimated before the input period.
    Missing returns are omitted, so the clock advances in observed return periods.
    """

    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be between zero and one")
    if not isinstance(calibration_window, (int, np.integer)) or calibration_window < 2:
        raise ValueError("calibration_window must be an integer of at least two")
    if initial_variance is not None and (
        not np.isfinite(initial_variance) or initial_variance < 0.0
    ):
        raise ValueError("initial_variance must be finite and nonnegative")
    clean = returns.astype(float).dropna()
    if not np.isfinite(clean).all():
        raise ValueError("returns must be finite")
    variance = pd.Series(np.nan, index=clean.index, dtype=float)
    start = calibration_window if initial_variance is None else 0
    if start < len(clean):
        variance.iloc[start] = (
            clean.iloc[:calibration_window].var(ddof=0)
            if initial_variance is None
            else initial_variance
        )
    for i in range(start + 1, len(clean)):
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
    horizon: int = 1,
    lambda_: float = 0.94,
    *,
    calibration_window: int = 20,
    initial_variance: float | None = None,
) -> pd.DataFrame:
    """Compare after-close-t normal VaR with compounded returns t+1 through t+h.

    The forecast incorporates the observed return at t. Calibration observations
    and incomplete forward windows are excluded. The zero-mean normal model uses
    square-root-of-time scaling, which is an approximation for compounded losses.
    For horizons above one, daily rows contain overlapping losses: applying an
    ordinary independent-Bernoulli Kupiec reference distribution to those rows is
    not justified. Even at horizon one, this test does not establish independence.
    """

    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    clean = returns.astype(float).dropna()
    prior_volatility = ewma_volatility(
        clean,
        lambda_=lambda_,
        calibration_window=calibration_window,
        initial_variance=initial_variance,
    )
    volatility = np.sqrt(lambda_ * prior_volatility**2 + (1.0 - lambda_) * clean**2)
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
    """Kupiec unconditional-coverage test with an asymptotic chi-square reference.

    Missing indicators are excluded. This reference assumes independent Bernoulli
    trials under the null; overlapping multi-period losses violate that assumption.
    The statistic tests the breach rate, not the independence of the indicators.
    """

    if not 0.0 < expected_probability < 1.0:
        raise ValueError("expected_probability must be between zero and one")
    clean = breaches.dropna()
    if not clean.isin([False, True]).all():
        raise ValueError("breaches must contain boolean or zero/one indicators")
    clean = clean.astype(bool)
    n = len(clean)
    x = int(clean.sum())
    if n == 0:
        raise ValueError("breaches cannot be empty")
    observed = x / n
    log_likelihood_expected = (n - x) * np.log1p(-expected_probability) + x * np.log(
        expected_probability
    )
    # At an empirical rate of zero or one, the maximized likelihood is one;
    # the limiting terms 0 * log(0) contribute zero, rather than NaN.
    log_likelihood_observed = 0.0
    if 0 < x < n:
        log_likelihood_observed = (n - x) * np.log(1.0 - observed) + x * np.log(observed)
    lr = max(0.0, 2.0 * (log_likelihood_observed - log_likelihood_expected))
    return {
        "observations": float(n),
        "breaches": float(x),
        "expected_breach_rate": float(expected_probability),
        "observed_breach_rate": float(observed),
        "lr_statistic": float(lr),
        "p_value": float(chi2.sf(lr, df=1)),
    }
