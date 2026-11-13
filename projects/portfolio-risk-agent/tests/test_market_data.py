"""No network: Yahoo responses are injected and fixtures are temporary files."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qfa.market_data import MarketDataError, load_market_data


NOW = datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)


def fixture(tmp_path, **kwargs):
    frame = pd.DataFrame({"Date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
                          "SYNTH_A": [100.0, 110.0, 99.0, 120.0],
                          "SYNTH_B": [100.0, 95.0, 95.0, 96.0]})
    for key, value in kwargs.items():
        frame[key] = value
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    return path


def load_fixture(tmp_path, path, **kwargs):
    arguments = {"source": "fixture", "tickers": ["SYNTH_A", "SYNTH_B"],
                 "start": "2020-01-02", "end": "2020-01-08",
                 "fixture_path": path, "output_dir": tmp_path / "out", "clock": lambda: NOW}
    return load_market_data(**{**arguments, **kwargs})


def test_fixture_returns_order_dates_and_hashes_are_auditable(tmp_path):
    path = fixture(tmp_path)
    data = load_fixture(tmp_path, path, tickers=["SYNTH_B", "SYNTH_A"], end="2020-01-07")
    assert list(data.prices.columns) == ["SYNTH_B", "SYNTH_A"]
    assert data.prices.index.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03", "2020-01-06"]
    np.testing.assert_allclose(data.returns.to_numpy(), [[-0.05, 0.1], [0.0, -0.1]], atol=1e-12)
    assert data.metadata["prices_sha256"] == hashlib.sha256(data.prices_path.read_bytes()).hexdigest()
    assert data.metadata["returns_sha256"] == hashlib.sha256(data.returns_path.read_bytes()).hexdigest()
    assert data.metadata["fixture_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data.metadata["fetched_at_utc"] == NOW.isoformat()
    assert data.metadata["synthetic"] is True
    assert data.prices_path.read_text().startswith("Date,SYNTH_B,SYNTH_A\n")
    json.dumps(data.summary(), allow_nan=False)
    repeated = load_fixture(tmp_path, path, tickers=["SYNTH_B", "SYNTH_A"], end="2020-01-07")
    assert repeated.dataset_id == data.dataset_id


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_bad_prices_are_rejected_without_forward_fill_or_output(tmp_path, bad):
    path = fixture(tmp_path, SYNTH_A=[100.0, bad, 99.0, 120.0])
    with pytest.raises(MarketDataError):
        load_fixture(tmp_path, path)
    assert not (tmp_path / "out").exists()


def test_duplicate_dates_missing_assets_and_malformed_requests_rejected(tmp_path):
    path = fixture(tmp_path, Date=["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-07"])
    with pytest.raises(MarketDataError, match="Duplicate observation"):
        load_fixture(tmp_path, path)
    path = fixture(tmp_path)
    with pytest.raises(MarketDataError, match="requested assets"):
        load_fixture(tmp_path, path, tickers=["MISSING"])
    for change in [{"tickers": ["../../private"]}, {"start": "2020-02-30"},
                   {"end": "2020-01-01"}, {"tickers": ["SYNTH_A", "SYNTH_A"]}]:
        with pytest.raises(ValueError):
            load_fixture(tmp_path, path, **change)


def test_yahoo_uses_adjusted_close_explicit_settings_and_requested_order(tmp_path):
    columns = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB"), ("Open", "AAA")])
    raw = pd.DataFrame([[100, 200, 99], [110, 198, 109], [121, 202, 120]],
                       index=pd.date_range("2024-01-02", periods=3, tz="America/New_York"), columns=columns)
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        return raw
    data = load_market_data(source="yahoo", tickers=["BBB", "AAA"], start="2024-01-01",
                            end="2024-01-05", output_dir=tmp_path, downloader=download, clock=lambda: NOW)
    assert list(data.prices.columns) == ["BBB", "AAA"]
    np.testing.assert_allclose(data.returns["AAA"], [0.1, 0.1])
    request = requests[0]
    for key, expected in {"auto_adjust": True, "actions": False, "repair": False,
                          "keepna": True, "interval": "1d", "threads": False,
                          "progress": False, "start": "2024-01-01", "end": "2024-01-05"}.items():
        assert request[key] == expected
    assert data.metadata["parameters"] == request
    assert data.metadata["synthetic"] is False
    assert data.prices.index.tz is None


@pytest.mark.parametrize("layout", ["single", "ticker_first"])
def test_yahoo_close_column_layouts(tmp_path, layout):
    columns = ["Close"] if layout == "single" else pd.MultiIndex.from_tuples([("AAA", "Close")])
    raw = pd.DataFrame([[100], [101], [102]], index=pd.date_range("2024-01-02", periods=3), columns=columns)
    data = load_market_data(source="yahoo", tickers=["AAA"], start="2024-01-01", end="2024-01-05",
                            output_dir=tmp_path, downloader=lambda **_: raw)
    assert list(data.prices.columns) == ["AAA"]


@pytest.mark.parametrize("missing", [True, False])
def test_yahoo_partial_asset_or_nan_response_does_not_silently_reduce_universe(tmp_path, missing):
    columns = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB")])
    raw = pd.DataFrame([[100, 200], [101, np.nan], [102, 201]],
                       index=pd.date_range("2024-01-02", periods=3), columns=columns)
    if missing:
        raw = raw.drop(columns=[("Close", "BBB")])
    with pytest.raises(MarketDataError):
        load_market_data(source="yahoo", tickers=["AAA", "BBB"], start="2024-01-01", end="2024-01-05",
                         output_dir=tmp_path / "out", downloader=lambda **_: raw)
    assert not (tmp_path / "out").exists()


def test_provider_error_is_safe_and_network_injection_is_used(tmp_path):
    def fail(**kwargs):
        raise RuntimeError("private provider credential and filesystem path")
    with pytest.raises(MarketDataError, match="provider request failed") as error:
        load_market_data(source="yahoo", tickers=["AAA"], start="2024-01-01", end="2024-01-05",
                         output_dir=tmp_path, downloader=fail)
    assert "private" not in str(error.value)


def test_optional_provider_cache_is_redirected_before_download(tmp_path, monkeypatch):
    recorded = []
    raw = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2024-01-02", periods=3))

    def download(**kwargs):
        assert recorded == [str(tmp_path / "provider_cache")]
        return raw
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(
        set_tz_cache_location=recorded.append, download=download,
    ))
    data = load_market_data(source="yahoo", tickers=["AAA"], start="2024-01-01", end="2024-01-05",
                            output_dir=tmp_path)
    assert data.metadata["provider"] == "Yahoo Finance via yfinance"
