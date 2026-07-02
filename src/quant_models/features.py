"""Feature engineering for equity-index trend models."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "intraday_return",
    "overnight_gap",
    "range_pct",
    "close_to_high",
    "close_to_low",
    "volume_change",
    "volume_z20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "ema_12_26",
    "rsi_14",
]


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Wilder-style RSI from a closing-price series."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_trend_dataset(
    ohlcv: pd.DataFrame,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a supervised next-day direction dataset.

    Features at date t are built from information available by that close.
    The target and backtest return are the close-to-close move from t to t+1.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = ohlcv.copy().sort_index()
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    returns = close.pct_change()

    frame["return_1d"] = returns
    frame["return_2d"] = close.pct_change(2)
    frame["return_5d"] = close.pct_change(5)
    frame["intraday_return"] = close / open_ - 1.0
    frame["overnight_gap"] = open_ / close.shift(1) - 1.0
    frame["range_pct"] = high / low - 1.0
    frame["close_to_high"] = close / high - 1.0
    frame["close_to_low"] = close / low - 1.0
    frame["volume_change"] = volume.pct_change()

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    frame["volume_z20"] = (volume - volume_mean) / volume_std

    for window in (5, 10, 20):
        frame[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252.0)
        frame[f"sma_{window}_ratio"] = close / close.rolling(window).mean() - 1.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    frame["ema_12_26"] = ema_12 / ema_26 - 1.0
    frame["rsi_14"] = relative_strength_index(close) / 100.0

    frame["forward_return"] = close.pct_change().shift(-1)
    frame["target_up"] = (frame["forward_return"] > threshold).astype(int)

    model_frame = frame.replace([np.inf, -np.inf], np.nan)
    model_frame = model_frame.dropna(subset=FEATURE_COLUMNS + ["forward_return", "target_up"])
    return model_frame, FEATURE_COLUMNS.copy()
