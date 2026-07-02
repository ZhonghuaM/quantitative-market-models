"""Time-series research tools for noisy market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def ar1_fit(series: pd.Series) -> dict[str, float]:
    """Fit a simple AR(1) model y_t = c + phi y_{t-1} + error."""

    clean = series.astype(float).dropna()
    if len(clean) < 3:
        raise ValueError("AR(1) fit requires at least three observations")
    y = clean.iloc[1:].to_numpy()
    x = clean.shift(1).dropna().to_numpy().reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    residual = y - model.predict(x)
    return {
        "intercept": float(model.intercept_),
        "phi": float(model.coef_[0]),
        "residual_volatility": float(residual.std(ddof=1)),
        "half_life": float(-np.log(2.0) / np.log(abs(model.coef_[0])))
        if 0.0 < abs(model.coef_[0]) < 1.0
        else np.nan,
    }


def exponential_smoothing(series: pd.Series, alpha: float = 0.2) -> pd.Series:
    """Single exponential smoothing with fixed alpha."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    clean = series.astype(float).dropna()
    smoothed = pd.Series(index=clean.index, dtype=float, name="exp_smoothing")
    smoothed.iloc[0] = clean.iloc[0]
    for i in range(1, len(clean)):
        smoothed.iloc[i] = alpha * clean.iloc[i] + (1.0 - alpha) * smoothed.iloc[i - 1]
    return smoothed


def kalman_local_level(
    series: pd.Series,
    observation_variance: float | None = None,
    state_variance: float | None = None,
) -> pd.DataFrame:
    """One-dimensional local-level Kalman filter."""

    clean = series.astype(float).dropna()
    if clean.empty:
        raise ValueError("series cannot be empty")
    observation_var = float(observation_variance or clean.var(ddof=1))
    state_var = float(state_variance or observation_var * 0.05)
    level = clean.iloc[0]
    covariance = observation_var
    records = []
    for date, value in clean.items():
        predicted_level = level
        predicted_covariance = covariance + state_var
        innovation = value - predicted_level
        innovation_variance = predicted_covariance + observation_var
        kalman_gain = predicted_covariance / innovation_variance
        level = predicted_level + kalman_gain * innovation
        covariance = (1.0 - kalman_gain) * predicted_covariance
        records.append(
            {
                "date": date,
                "observation": value,
                "filtered_level": level,
                "innovation": innovation,
                "kalman_gain": kalman_gain,
            }
        )
    return pd.DataFrame(records).set_index("date")


def pairs_spread_signal(
    asset: pd.Series,
    hedge: pd.Series,
    window: int = 60,
    entry_z: float = 2.0,
) -> pd.DataFrame:
    """Estimate rolling hedge-ratio spread and z-score pair signals."""

    frame = pd.concat([asset.rename("asset"), hedge.rename("hedge")], axis=1).dropna()
    if len(frame) < window + 2:
        raise ValueError("not enough observations for rolling pair signal")

    hedge_ratio = pd.Series(index=frame.index, dtype=float)
    spread = pd.Series(index=frame.index, dtype=float)
    for i in range(window, len(frame)):
        sample = frame.iloc[i - window : i]
        model = LinearRegression().fit(sample[["hedge"]], sample["asset"])
        hedge_ratio.iloc[i] = model.coef_[0]
        spread.iloc[i] = frame["asset"].iloc[i] - model.predict(frame[["hedge"]].iloc[[i]])[0]

    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    signal = pd.Series(0.0, index=frame.index, name="pair_signal")
    signal.loc[zscore < -entry_z] = 1.0
    signal.loc[zscore > entry_z] = -1.0
    return pd.DataFrame(
        {
            "hedge_ratio": hedge_ratio,
            "spread": spread,
            "zscore": zscore,
            "signal": signal,
        }
    ).dropna()


def regime_transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Estimate transition frequencies between discrete regime labels."""

    clean = labels.dropna().astype(str)
    transitions = pd.crosstab(clean.shift(1), clean, normalize="index")
    transitions.index.name = "from_regime"
    transitions.columns.name = "to_regime"
    return transitions.fillna(0.0)
