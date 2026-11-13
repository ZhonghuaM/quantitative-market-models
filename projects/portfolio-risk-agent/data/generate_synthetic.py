#!/usr/bin/env python3
"""Generate the project's fully synthetic, explicitly non-market price fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np


ASSETS = ["SYNTH_EQUITY", "SYNTH_DEFENSIVE", "SYNTH_BOND"]
SEED = 20261113
ROWS = 900
START = date(2020, 1, 2)
ANNUAL_LOG_DRIFT = np.array([0.07, 0.025, 0.035])
ANNUAL_VOLATILITY = np.array([0.16, 0.08, 0.05])
CORRELATION = np.array([[1.0, 0.25, -0.15], [0.25, 1.0, 0.20], [-0.15, 0.20, 1.0]])
REGIMES = [
    {"first_return_index": 0, "stop_return_index": 300, "volatility_multiplier": 1.0, "annual_log_drift_shift": [0.0, 0.0, 0.0]},
    {"first_return_index": 300, "stop_return_index": 600, "volatility_multiplier": 1.35, "annual_log_drift_shift": [-0.02, -0.005, 0.005]},
    {"first_return_index": 600, "stop_return_index": 899, "volatility_multiplier": 0.85, "annual_log_drift_shift": [0.01, 0.005, 0.0]},
]


def weekday_dates(start: date, count: int) -> list[date]:
    """Monday–Friday labels only; deliberately not an exchange calendar."""
    dates = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def generate(output: Path) -> dict:
    rng = np.random.Generator(np.random.PCG64(SEED))
    covariance = np.outer(ANNUAL_VOLATILITY, ANNUAL_VOLATILITY) * CORRELATION / 252.0
    innovations = rng.standard_normal((ROWS - 1, len(ASSETS))) @ np.linalg.cholesky(covariance).T
    log_returns = np.empty_like(innovations)
    for regime in REGIMES:
        selection = slice(regime["first_return_index"], regime["stop_return_index"])
        drift = (ANNUAL_LOG_DRIFT + regime["annual_log_drift_shift"]) / 252.0
        log_returns[selection] = drift + regime["volatility_multiplier"] * innovations[selection]
    prices = np.vstack([np.full(len(ASSETS), 100.0), 100.0 * np.exp(np.cumsum(log_returns, axis=0))])
    dates = weekday_dates(START, ROWS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Date", *ASSETS])
        writer.writerows([day.isoformat(), *[format(float(x), ".17g") for x in row]] for day, row in zip(dates, prices))
    manifest = {
        "schema_version": 1,
        "data_kind": "fully synthetic; no observations sourced from a financial market",
        "provider": "repository generator data/generate_synthetic.py",
        "licence": "MIT; generated for this repository",
        "file": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "rows": ROWS,
        "return_observations": ROWS - 1,
        "assets": ASSETS,
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "date_labels": "Monday–Friday only; not an exchange calendar; public holidays are included",
        "timezone": None,
        "timezone_note": "Date-only synthetic labels; no trading timestamps or real venue",
        "currency": "arbitrary synthetic price units",
        "initial_prices": [100.0, 100.0, 100.0],
        "seed": SEED,
        "rng": "NumPy Generator(PCG64)",
        "process": "correlated Gaussian log increments, deterministic piecewise drift/volatility; positive prices by cumulative exponentiation",
        "annualization_factor": 252,
        "annual_log_drift": ANNUAL_LOG_DRIFT.tolist(),
        "annual_volatility": ANNUAL_VOLATILITY.tolist(),
        "correlation": CORRELATION.tolist(),
        "regimes": REGIMES,
        "regime_index_note": "Return index 0 connects price rows 0 and 1; stop indices are exclusive",
        "adjustments": "none; no dividends, splits, missing rows, fees, or other corporate actions are simulated",
        "purpose": "deterministic software/numerical evaluation, not evidence of investment performance",
        "reproduction_note": "Committed bytes are canonical. Generator records NumPy version; floating-point/library differences may affect final digits.",
    }
    output.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("synthetic_prices.csv"))
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps({key: result[key] for key in ("file", "rows", "start", "end", "sha256")}, indent=2))
