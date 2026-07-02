"""Data loading helpers for bundled market datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: column.strip().lower().replace(" ", "_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def load_ohlcv(path: str | Path | None = None) -> pd.DataFrame:
    """Load an OHLCV dataset with a parsed, sorted datetime index."""

    csv_path = Path(path) if path is not None else DATA_DIR / "vas_equity_etf.csv"
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    frame = _normalise_columns(frame)
    if "date" not in frame:
        raise ValueError(f"{csv_path} must contain a Date column")

    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True)
    frame = frame.sort_values("date").set_index("date")
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return frame


def load_price_series(
    path: str | Path | None = None,
    price_column: str = "SP500",
) -> pd.Series:
    """Load a single price series with a parsed datetime index."""

    csv_path = Path(path) if path is not None else DATA_DIR / "sp500_index.csv"
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Date" not in frame:
        raise ValueError(f"{csv_path} must contain a Date column")
    if price_column not in frame:
        raise ValueError(f"{csv_path} must contain a {price_column!r} column")

    dates = pd.to_datetime(frame["Date"], dayfirst=True)
    series = pd.Series(frame[price_column].astype(float).values, index=dates, name=price_column)
    return series.sort_index()


def close_to_close_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series to simple close-to-close returns."""

    returns = prices.astype(float).pct_change().dropna()
    returns.name = f"{prices.name or 'price'}_return"
    return returns
