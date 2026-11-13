"""Validated daily price ingestion with explicit provenance and no imputation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import validate_tool_arguments


class MarketDataError(ValueError):
    """Market data cannot support the requested calculation."""


@dataclass(frozen=True)
class MarketDataset:
    dataset_id: str
    prices: pd.DataFrame
    returns: pd.DataFrame
    metadata: dict
    prices_path: Path
    returns_path: Path

    def summary(self) -> dict:
        """Small JSON payload for the agent; full observations remain in files."""
        return {"dataset_id": self.dataset_id, "tickers": list(self.prices.columns),
                "price_observations": len(self.prices), "return_observations": len(self.returns),
                "first_date": self.metadata["first_date"], "last_date": self.metadata["last_date"],
                "provider": self.metadata["provider"], "synthetic": self.metadata["synthetic"],
                "price_sha256": self.metadata["prices_sha256"],
                "provenance_file": f"{self.dataset_id}_metadata.json"}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index_label="Date", date_format="%Y-%m-%d",
                        float_format="%.17g", lineterminator="\n").encode("utf-8")


def _yahoo_closes(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise MarketDataError("The provider returned no observations.")
    if isinstance(raw.columns, pd.MultiIndex):
        close_levels = [level for level in range(raw.columns.nlevels)
                        if "Close" in raw.columns.get_level_values(level)]
        if len(close_levels) != 1:
            raise MarketDataError("The provider response has no unambiguous adjusted Close field.")
        prices = raw.xs("Close", axis=1, level=close_levels[0])
        if isinstance(prices.columns, pd.MultiIndex):
            raise MarketDataError("The provider response has an unsupported column layout.")
        return prices
    if len(tickers) == 1 and "Close" in raw.columns:
        return raw[["Close"]].rename(columns={"Close": tickers[0]})
    raise MarketDataError("The provider response does not identify all requested assets.")


def _validate_prices(prices: pd.DataFrame, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if prices.columns.has_duplicates:
        raise MarketDataError("Duplicate asset columns are not permitted.")
    if any(ticker not in prices.columns for ticker in tickers):
        raise MarketDataError("One or more requested assets are missing from the data.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="raise"))
    except (ValueError, TypeError):
        raise MarketDataError("Data contain invalid observation dates.") from None
    if index.isna().any():
        raise MarketDataError("Data contain missing observation dates.")
    # Daily provider dates identify local sessions, not UTC timestamps.
    if index.tz is not None:
        index = index.tz_localize(None)
    if not index.equals(index.normalize()):
        raise MarketDataError("Daily observations must use dates without intraday timestamps.")
    if index.has_duplicates:
        raise MarketDataError("Duplicate observation dates are not permitted.")
    prices = prices.loc[:, tickers].copy()
    prices.index = index
    prices = prices.sort_index()
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end))]
    if len(prices) < 3:
        raise MarketDataError("At least three price observations are required in the requested date range.")
    try:
        prices = prices.astype(float)
    except (TypeError, ValueError):
        raise MarketDataError("Prices must be numeric.") from None
    if not np.isfinite(prices.to_numpy()).all():
        raise MarketDataError("Missing or nonfinite prices are rejected; prices are never forward-filled.")
    if (prices.to_numpy() <= 0).any():
        raise MarketDataError("Prices must be strictly positive.")
    prices.index.name = "Date"
    prices.columns.name = None
    return prices


def load_market_data(*, source: str, tickers: list[str], start: str, end: str,
                     output_dir: str | Path, fixture_path: str | Path | None = None,
                     downloader: Callable | None = None,
                     clock: Callable[[], datetime] | None = None) -> MarketDataset:
    """Load a local fixture or explicit Yahoo adjusted daily close request.

    The application, not the model, supplies fixture_path and output_dir. Yahoo
    is optional and imported only for a live request without an injected test
    downloader. No missing observations are filled or silently dropped.
    """
    validate_tool_arguments("load_market_data", {
        "source": source, "tickers": tickers, "start": start, "end": end,
    })
    fixture_hash = None
    request = {"tickers": list(tickers), "start": start, "end": end,
               "interval": "1d", "auto_adjust": True, "actions": False,
               "repair": False, "keepna": True, "threads": False, "progress": False,
               "group_by": "column", "ignore_tz": True, "timeout": 20}
    if source == "fixture":
        if fixture_path is None:
            raise MarketDataError("The application must supply a fixture file.")
        path = Path(fixture_path)
        fixture_bytes = path.read_bytes()
        fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        raw = pd.read_csv(path)
        date_columns = [column for column in raw.columns if str(column).lower() == "date"]
        if len(date_columns) != 1:
            raise MarketDataError("The fixture must contain exactly one Date column.")
        prices = raw.set_index(date_columns[0])
        provider = "local_synthetic_fixture"
        request = {"tickers": list(tickers), "start": start, "end": end}
    else:
        if downloader is None:
            try:
                import yfinance
            except ImportError:
                raise MarketDataError("Yahoo downloads require the optional yfinance dependency.") from None
            # yfinance also uses this location for its cookie cache. Keep all
            # provider state inside the application-owned run directory.
            yfinance.set_tz_cache_location(str(Path(output_dir) / "provider_cache"))
            downloader = yfinance.download
        try:
            raw = downloader(**request)
        except Exception:
            raise MarketDataError("The market-data provider request failed.") from None
        prices = _yahoo_closes(raw, tickers)
        provider = "Yahoo Finance via yfinance"
    prices = _validate_prices(prices, tickers, start, end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise MarketDataError("Computed returns are nonfinite.")
    prices_bytes, returns_bytes = _canonical_csv(prices), _canonical_csv(returns)
    prices_hash = hashlib.sha256(prices_bytes).hexdigest()
    identity = json.dumps({"source": source, "request": request, "prices_sha256": prices_hash},
                          sort_keys=True).encode()
    dataset_id = "data_" + hashlib.sha256(identity).hexdigest()[:16]
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if now.tzinfo is None:
        raise MarketDataError("The provenance clock must return a timezone-aware time.")
    metadata = {
        "schema_version": 1, "dataset_id": dataset_id, "source": source,
        "provider": provider, "synthetic": source == "fixture",
        "fetched_at_utc": now.astimezone(timezone.utc).isoformat(),
        "parameters": request, "date_bounds": "start inclusive, end exclusive",
        "tickers": list(tickers), "first_date": prices.index[0].date().isoformat(),
        "last_date": prices.index[-1].date().isoformat(),
        "price_observations": len(prices), "return_observations": len(returns),
        "return_definition": "simple close-to-close returns; pct_change(fill_method=None)",
        "prices_sha256": prices_hash,
        "returns_sha256": hashlib.sha256(returns_bytes).hexdigest(),
        "fixture_sha256": fixture_hash,
        "versions": {name: _version(name) for name in ("numpy", "pandas", "yfinance")},
        "adjustments": "synthetic prices, no corporate actions" if source == "fixture"
                       else "yfinance auto_adjust=True Close; Yahoo adjustment history may be revised",
        "calendar_timezone": "daily session-date labels; exchange-calendar completeness is not verified",
        "missing_observations": "any missing/nonfinite requested asset price on a returned row is rejected; no filling",
        "licensing": "repository synthetic fixture" if source == "fixture"
                     else "User-supplied Yahoo request; provider terms apply; downloaded data are not bundled for redistribution",
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prices_path = directory / f"{dataset_id}_prices.csv"
    returns_path = directory / f"{dataset_id}_returns.csv"
    prices_path.write_bytes(prices_bytes)
    returns_path.write_bytes(returns_bytes)
    (directory / f"{dataset_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return MarketDataset(dataset_id, prices, returns, metadata, prices_path, returns_path)
