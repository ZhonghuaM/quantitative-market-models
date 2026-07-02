#!/usr/bin/env python
"""Download reproducible public datasets for optional experiments.

Bundled sample data is enough to run the repository. This script is provided so
users can refresh or extend datasets without embedding credentials in the code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_AGENT = "quantitative-market-models/0.2 contact: public-github-demo@example.com"


def download_stooq_daily(symbol: str, output: Path) -> Path:
    """Download daily OHLCV data from Stooq's public CSV endpoint."""

    query = urlencode({"s": symbol.lower(), "i": "d"})
    url = f"https://stooq.com/q/d/l/?{query}"
    frame = pd.read_csv(url)
    if frame.empty or "Date" not in frame:
        raise RuntimeError(f"No Stooq data returned for {symbol}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def download_sec_company_facts(cik: str, output: Path, user_agent: str) -> Path:
    """Download SEC company-facts JSON for a zero-padded CIK."""

    normalized = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized}.json"
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    data = json.loads(payload.decode("utf-8"))
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    time.sleep(0.2)
    return output


def download_fred_series(series_id: str, output: Path) -> Path:
    """Download a public FRED graph CSV without requiring an API key."""

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id.upper()}"
    frame = pd.read_csv(url)
    if frame.empty or "DATE" not in frame:
        raise RuntimeError(f"No FRED data returned for {series_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    market = subparsers.add_parser("market", help="Download daily Stooq market data")
    market.add_argument("--symbol", default="spy.us", help="Stooq symbol such as spy.us or ^spx")
    market.add_argument("--output", type=Path, default=ROOT / "data" / "downloaded" / "spy_stooq.csv")

    sec = subparsers.add_parser("sec-facts", help="Download SEC company facts JSON")
    sec.add_argument("--cik", default="0000320193", help="Company CIK, with or without leading zeros")
    sec.add_argument("--output", type=Path, default=ROOT / "data" / "downloaded" / "company_facts.json")
    sec.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    fred = subparsers.add_parser("fred", help="Download a public FRED graph CSV")
    fred.add_argument("--series", default="DGS10", help="FRED series id, for example DGS10")
    fred.add_argument("--output", type=Path, default=ROOT / "data" / "downloaded" / "fred_dgs10.csv")

    args = parser.parse_args(argv)
    if args.command == "market":
        path = download_stooq_daily(args.symbol, args.output)
    elif args.command == "sec-facts":
        path = download_sec_company_facts(args.cik, args.output, args.user_agent)
    elif args.command == "fred":
        path = download_fred_series(args.series, args.output)
    else:
        raise AssertionError(args.command)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
