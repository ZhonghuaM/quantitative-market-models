"""Macroeconomic data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fred_graph_csv_url(series_id: str) -> str:
    """Return a no-key FRED graph CSV URL for a public series id."""

    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id.upper()}"


def load_fred_csv(path: str | Path, series_id: str | None = None) -> pd.Series:
    """Load a FRED graph CSV file with DATE and series columns."""

    frame = pd.read_csv(path)
    if "DATE" not in frame:
        raise ValueError("FRED CSV must contain DATE")
    value_column = series_id or next(column for column in frame.columns if column != "DATE")
    series = pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").values,
        index=pd.to_datetime(frame["DATE"]),
        name=value_column,
    )
    return series.dropna().sort_index()


def align_macro_to_market(market: pd.DataFrame, macro: pd.Series) -> pd.DataFrame:
    """Forward-fill a lower-frequency macro series onto market dates."""

    aligned = macro.reindex(market.index.union(macro.index)).sort_index().ffill().reindex(market.index)
    output = market.copy()
    output[macro.name or "macro"] = aligned
    return output
