#!/usr/bin/env python3
"""Run an actual native-versus-independent numerical evaluation, without an LLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import scipy

from reference import historical_tail, kupiec_reference, lognormal_moments, optimise_reference, read_prices, risk_metrics, simple_returns


ROOT = Path(__file__).resolve().parents[1]


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def close(self, name: str, actual, reference, *, atol: float = 1e-10, rtol: float = 1e-8) -> None:
        actual_array, expected_array = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
        same_shape = actual_array.shape == expected_array.shape
        passed = same_shape and bool(np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=False))
        difference = float(np.max(np.abs(actual_array - expected_array))) if same_shape and actual_array.size else None
        self.rows.append({
            "check": name, "passed": passed, "actual": actual_array.tolist(), "reference": expected_array.tolist(),
            "max_absolute_error": difference, "absolute_tolerance": atol, "relative_tolerance": rtol,
        })

    def condition(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})


def invoke(engine: Path, command: str, prices: Path, weights: list[float], confidence: float, **options) -> dict:
    argv = [str(engine), command, "--prices", str(prices), "--weights", ",".join(map(str, weights)), "--confidence", str(confidence)]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def write_prices(path: Path, assets: list[str], dates: list[str], prices: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Date", *assets])
        writer.writerows([day, *[format(float(x), ".17g") for x in row]] for day, row in zip(dates, prices))


def scalar_csv(path: Path, values: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(values.items())


def evaluate(engine: Path, output: Path, prices_path: Path | None = None) -> dict:
    engine = engine.resolve()
    prices_path = (prices_path or ROOT / "data/synthetic_prices.csv").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/demo.json").read_text())
    weights = config["weights"]
    confidence = config["confidence"]
    assets, dates, prices = read_prices(prices_path)
    returns = simple_returns(prices)
    portfolio = returns @ np.array(weights)
    checks = Checks()
    native_outputs = {}
    try:
        risk = invoke(engine, "risk", prices_path, weights, confidence)
        native_outputs["risk"] = risk
        references = risk_metrics(portfolio, confidence)
        for name, expected in references.items():
            checks.close("historical_" + name, risk[name], expected)
        checks.close("asset_sample_means", risk["asset_mean_daily_returns"], returns.mean(axis=0), atol=1e-12)
        checks.close("sample_covariance_ddof1", risk["sample_covariance"], np.cov(returns, rowvar=False, ddof=1), atol=1e-12)

        # These exports preserve native outputs for an independent base-R check.
        scalar_csv(output / "native_risk_summary.csv", {
            **{name: risk[name] for name in references}, "confidence": confidence,
            **{f"weight_{asset}": weight for asset, weight in zip(assets, weights)},
        })
        with (output / "native_covariance.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["asset", *assets])
            writer.writerows([asset, *row] for asset, row in zip(assets, risk["sample_covariance"]))
        with (output / "daily_portfolio_returns.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["date", "reference_portfolio_return"])
            writer.writerows(zip(dates[1:], portfolio))

        training_count = config["train_observations"]
        training = returns[:training_count]
        optimisation = invoke(engine, "optimise", prices_path, weights, confidence, train=training_count, frontier_points=config["frontier_points"])
        native_outputs["optimisation"] = optimisation
        checks.close("optimisation_daily_covariance_ridge", optimisation["ridge"], 1e-8, atol=0, rtol=0)
        checks.close("optimisation_training_sample_covariance", optimisation["sample_covariance"], np.cov(training, rowvar=False, ddof=1), atol=1e-12)
        independent_weights, independent_variance = optimise_reference(training)
        minimum = optimisation["minimum_variance"]
        checks.close("minimum_variance_weights_scipy", minimum["weights"], independent_weights, atol=2e-6)
        checks.close("minimum_variance_objective_scipy", minimum["estimated_daily_variance"], independent_variance, atol=1e-11)
        mean = training.mean(axis=0)
        covariance = np.cov(training, rowvar=False, ddof=1) + np.eye(len(assets)) * 1e-8
        checks.condition("frontier_row_count", len(optimisation["frontier"]) == config["frontier_points"], "Native rows match the requested frontier resolution.")
        for i, point in enumerate(optimisation["frontier"]):
            allocation = np.asarray(point["weights"])
            target = point["target_daily_return"]
            _, independent_point_variance = optimise_reference(training, target)
            checks.close(f"frontier_{i:02d}_sum_weights", allocation.sum(), 1.0, atol=1e-9)
            checks.condition(f"frontier_{i:02d}_long_only", bool(np.all(allocation >= -1e-9) and np.all(allocation <= 1 + 1e-9)), "All weights within [0, 1] to numerical tolerance.")
            checks.close(f"frontier_{i:02d}_target_return", mean @ allocation, target, atol=1e-10)
            checks.close(f"frontier_{i:02d}_reported_variance", point["estimated_daily_variance"], allocation @ covariance @ allocation, atol=1e-12)
            checks.close(f"frontier_{i:02d}_variance_scipy", point["estimated_daily_variance"], independent_point_variance, atol=1e-11)
        for sample, rows in [("training", training), ("holdout", returns[training_count:])]:
            expected_metrics = risk_metrics(rows @ np.asarray(minimum["weights"]), confidence)
            for name in ("mean_daily_return", "volatility_daily", "historical_var", "historical_es"):
                checks.close(f"minimum_variance_{sample}_{name}", optimisation["comparison"]["minimum_variance"][sample][name], expected_metrics[name])
            for label in ("input", "minimum_variance"):
                allocation = np.asarray(optimisation["comparison"][label]["weights"])
                contributions = optimisation["comparison"][label][sample + "_variance_contributions"]
                portfolio_variance = float(np.var(rows @ allocation, ddof=1))
                # Differentiate the variance of raw portfolio returns numerically,
                # without calling the native covariance/contribution calculation.
                step = 1e-5
                marginal_contributions = []
                for asset in range(len(allocation)):
                    perturbation = np.eye(len(allocation))[asset] * step
                    plus = np.var(rows @ (allocation + perturbation), ddof=1)
                    minus = np.var(rows @ (allocation - perturbation), ddof=1)
                    marginal_contributions.append(allocation[asset] * (plus - minus) / (4 * step))
                prefix = f"{label}_{sample}_variance_contributions"
                checks.close(prefix + "_finite_difference", contributions["absolute_daily_variance"], marginal_contributions, atol=1e-11)
                checks.close(prefix + "_sum_raw_variance", sum(contributions["absolute_daily_variance"]), portfolio_variance, atol=1e-12)
                checks.close(prefix + "_fractions", contributions["fractional"], np.asarray(marginal_contributions) / portfolio_variance, atol=1e-8)
                checks.close(prefix + "_fraction_sum", sum(contributions["fractional"]), 1.0, atol=1e-10)

        input_holdout = returns[training_count:] @ np.asarray(weights)
        candidate_holdout = returns[training_count:] @ np.asarray(minimum["weights"])
        input_var, input_es = historical_tail(input_holdout, confidence)
        candidate_var, candidate_es = historical_tail(candidate_holdout, confidence)
        expected_differences = {
            "annualized_variance": 252 * (np.var(input_holdout, ddof=1) - np.var(candidate_holdout, ddof=1)),
            "historical_var": input_var - candidate_var,
            "historical_es": input_es - candidate_es,
        }
        intervals = optimisation["holdout_uncertainty"]
        expected_keys = {(name, block) for name in expected_differences for block in (10, 20)}
        checks.condition("holdout_uncertainty_metric_blocks", {(row["metric"], row["block_size"]) for row in intervals} == expected_keys and len(intervals) == 6, "Three metrics at two block lengths are reported exactly once each.")
        for row in intervals:
            key = f"bootstrap_{row['metric']}_block_{row['block_size']}"
            checks.close(key + "_empirical_point", row["estimate"], expected_differences[row["metric"]], atol=1e-12)
            checks.condition(key + "_ordered_interval", math.isfinite(row["ci_lower"]) and math.isfinite(row["ci_upper"]) and row["ci_lower"] <= row["ci_upper"], "Finite lower and upper percentile endpoints are ordered; coverage is not inferred from this numerical check.")

        backtest = invoke(engine, "backtest", prices_path, weights, confidence, window=config["window"])
        native_outputs["backtest"] = backtest
        forecasts = backtest["forecasts"]
        window = config["window"]
        expected_var = [historical_tail(portfolio[i-window:i], confidence)[0] for i in range(window, len(portfolio))]
        checks.close("rolling_var_past_window_only", [row["var"] for row in forecasts], expected_var, atol=1e-12)
        checks.close("rolling_realized_loss_alignment", [row["loss"] for row in forecasts], -portfolio[window:], atol=1e-12)
        checks.condition("rolling_date_alignment", [row["date"] for row in forecasts] == dates[window+1:], "Forecast dates identify the realized return's ending date.")
        expected_breaches = (-portfolio[window:]) > np.asarray(expected_var)
        checks.close("rolling_breach_indicators", [row["breach"] for row in forecasts], expected_breaches.astype(int), atol=0, rtol=0)
        for key, value in kupiec_reference(int(expected_breaches.sum()), len(expected_breaches), 1-confidence).items():
            checks.close("kupiec_observed_" + key, backtest["kupiec"][key], value, atol=1e-12)

        with tempfile.TemporaryDirectory(prefix="qfa-independent-") as scratch:
            scratch_path = Path(scratch)
            # A training design whose constrained optimum is the second asset,
            # followed by a holdout in which it improves EVERY return by 0.002.
            # Pairing then makes every bootstrap tail-risk improvement constant;
            # variance is translation invariant. Independent resampling would fail.
            training_base = np.tile([-0.01, 0.01], 50)
            training_pair = np.column_stack([2 * training_base, training_base])
            holdout_base = 0.003 * np.cos(np.arange(60))
            shifted_pair = np.column_stack([holdout_base, holdout_base + 0.002])
            translation_returns = np.vstack([training_pair, shifted_pair])
            translation_prices = np.vstack([np.array([100.0, 100.0]), 100 * np.cumprod(1 + translation_returns, axis=0)])
            translation_dates = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(len(translation_prices))]
            translation_path = scratch_path / "constant_paired_improvement.csv"
            write_prices(translation_path, ["INPUT", "IMPROVED"], translation_dates, translation_prices)
            translation_result = invoke(engine, "optimise", translation_path, [1.0, 0.0], confidence, train=100, frontier_points=3)
            checks.close("bootstrap_translation_case_allocation", translation_result["minimum_variance"]["weights"], [0.0, 1.0], atol=1e-10)
            for row in translation_result["holdout_uncertainty"]:
                expected = 0.0 if row["metric"] == "annualized_variance" else 0.002
                for field in ("estimate", "ci_lower", "ci_upper"):
                    checks.close(f"paired_constant_improvement_{row['metric']}_block_{row['block_size']}_{field}", row[field], expected, atol=1e-12)

            altered = prices.copy()
            altered[-1] *= np.array([0.70, 1.20, 0.85])
            altered_path = scratch_path / "future_perturbation.csv"
            write_prices(altered_path, assets, dates, altered)
            future_backtest = invoke(engine, "backtest", altered_path, weights, confidence, window=window)
            checks.condition("future_edit_preserves_earlier_backtest", future_backtest["forecasts"][:-1] == forecasts[:-1], "Only the last price row was changed; all previous forecast/realization rows must be identical.")
            checks.close("future_edit_preserves_final_var", future_backtest["forecasts"][-1]["var"], forecasts[-1]["var"], atol=0, rtol=0)
            altered_optimisation = invoke(engine, "optimise", altered_path, weights, confidence, train=training_count, frontier_points=config["frontier_points"])
            checks.condition("holdout_edit_preserves_allocation", altered_optimisation["minimum_variance"] == minimum, "Training-only allocation cannot change when only a holdout price changes.")

            # Boundary coverage checks are executable end-to-end price inputs.
            for label, increments, expected_rate in [
                ("zero", np.zeros(1100), 0),
                ("all", -0.001 * np.arange(1, 81), 1),
            ]:
                boundary_prices = np.r_[100.0, 100.0 * np.cumprod(1 + increments)][:, None]
                boundary_dates = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(len(boundary_prices))]
                boundary_path = scratch_path / f"kupiec_{label}.csv"
                write_prices(boundary_path, ["BOUNDARY"], boundary_dates, boundary_prices)
                boundary_window = 100 if label == "zero" else 20
                result = invoke(engine, "backtest", boundary_path, [1.0], confidence, window=boundary_window)
                count = len(increments) - boundary_window
                expected_count = expected_rate * count
                checks.close(f"kupiec_{label}_breach_count", result["kupiec"]["breaches"], expected_count, atol=0, rtol=0)
                for key, value in kupiec_reference(expected_count, count, 1-confidence).items():
                    checks.close(f"kupiec_{label}_{key}", result["kupiec"][key], value, atol=1e-300 if key == "p_value" else 1e-9, rtol=1e-7)

            # The independent RNG need not match C: compare to analytic moments,
            # with tolerances derived from Monte Carlo sampling variation.
            log_increments = 0.0003 + 0.008 * np.cos(np.arange(1, 241) * np.sqrt(2.0))
            single_prices = np.r_[100.0, 100.0 * np.exp(np.cumsum(log_increments))][:, None]
            single_dates = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(len(single_prices))]
            single_path = scratch_path / "one_asset_lognormal.csv"
            write_prices(single_path, ["LOGNORMAL_CHECK"], single_dates, single_prices)
            paths, horizon = 50000, 5
            mc_args = {"paths": paths, "horizon": horizon, "seed": 42}
            simulation = invoke(engine, "monte-carlo", single_path, [1.0], confidence, **mc_args)
            repeated = invoke(engine, "monte-carlo", single_path, [1.0], confidence, **mc_args)
            checks.condition("monte_carlo_seeded_rerun", simulation == repeated, "Two native runs with identical inputs and seed return identical JSON values.")
            actual_log = np.log(single_prices[1:, 0] / single_prices[:-1, 0])
            expected_mean, expected_variance, fourth = lognormal_moments(float(actual_log.mean()), float(actual_log.var(ddof=1) + 1e-12), horizon)
            mean_se = math.sqrt(expected_variance / paths)
            variance_se = math.sqrt((fourth - ((paths - 3) / (paths - 1)) * expected_variance**2) / paths)
            checks.close("monte_carlo_analytic_mean_six_se", simulation["mean_simulated_return"], expected_mean, atol=6 * mean_se, rtol=0)
            checks.close("monte_carlo_analytic_variance_six_se", simulation["simulated_return_std"]**2, expected_variance, atol=6 * variance_se, rtol=0)
            checks.close("monte_carlo_covariance_ridge", simulation["covariance_ridge"], 1e-12, atol=0, rtol=0)
            native_outputs["analytic_monte_carlo_check"] = {
                "paths": paths, "horizon": horizon, "seed": 42,
                "mean_sampling_standard_error": mean_se, "variance_sampling_standard_error": variance_se,
                "interpretation": "Six-standard-error numerical sampling bounds, not investment-performance confidence intervals.",
            }

        simulation = invoke(engine, "monte-carlo", prices_path, weights, confidence, paths=config["paths"], horizon=config["horizon"], seed=config["seed"])
        native_outputs["fixture_monte_carlo"] = simulation
        log_returns = np.log(prices[1:] / prices[:-1])
        checks.close("monte_carlo_fitted_log_mean", simulation["estimated_log_mean"], log_returns.mean(axis=0), atol=1e-12)
        checks.close("monte_carlo_fitted_log_covariance", simulation["estimated_log_covariance"], np.cov(log_returns, rowvar=False, ddof=1), atol=1e-12)
        checks.close("monte_carlo_effective_covariance", simulation["effective_log_covariance"], np.cov(log_returns, rowvar=False, ddof=1) + np.eye(len(assets)) * 1e-12, atol=1e-14)
    except Exception as error:
        detail = str(error).replace(str(ROOT), "<project>").replace(tempfile.gettempdir(), "<temporary>")
        checks.condition("evaluation_execution", False, f"{type(error).__name__}: {detail}")

    report = {
        "schema_version": 1,
        "evaluation_kind": "executed independent numerical reference checks; no LLM calls",
        "passed": all(row["passed"] for row in checks.rows),
        "checks_passed": sum(row["passed"] for row in checks.rows),
        "checks_total": len(checks.rows),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        "reference_source_sha256": hashlib.sha256(Path(__file__).with_name("reference.py").read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "checks": checks.rows,
        "llm_tool_selection": {"status": "not_run", "reason": "This evaluator checks the numerical engine. Hand-authored contract cases are not held-out LLM accuracy evidence."},
        "r_independent_check": {"status": "not_run_by_this_python_evaluator", "command": "Rscript evaluation/check_reference.R --prices data/synthetic_prices.csv --output-dir <evaluation-output-dir>"},
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "native_outputs.json").write_text(json.dumps(native_outputs, indent=2, allow_nan=False) + "\n")
    with (output / "reference_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "passed", "max_absolute_error", "absolute_tolerance", "relative_tolerance", "actual", "reference", "detail"], lineterminator="\n")
        writer.writeheader()
        for row in checks.rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    lines = ["# Independent numerical evaluation", "", f"Result: **{report['checks_passed']}/{report['checks_total']} checks passed**.", "", "These checks execute the native binary and compare its outputs with NumPy/SciPy calculations. The input prices are fully synthetic. No real funds are managed, and these numerical checks do not establish investment performance.", "", "| Check | Result | Maximum absolute error |", "|---|---|---:|"]
    for row in checks.rows:
        error = row.get("max_absolute_error")
        lines.append(f"| {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {format(error, '.6g') if error is not None else '—'} |")
    lines.extend(["", "The optimizer reference is an independent SciPy SLSQP solve with the same explicit 1e-8 daily covariance ridge. Each frontier point is checked for feasibility and variance. Monte Carlo moment tolerances use six sampling standard errors against a one-asset analytical lognormal distribution; they are not trading-performance intervals.", "", "`native_risk_summary.csv` and `native_covariance.csv` export actual native values for the separate base-R checker. This Python report does not claim the R check was executed. `evaluation_report.json` records versions and hashes.", "", "LLM tool-selection accuracy was not evaluated by this command. Hand-authored tool-contract cases and scripted workflows must not be reported as held-out model accuracy.", ""])
    (output / "evaluation_report.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=ROOT / "data/synthetic_prices.csv")
    args = parser.parse_args()
    result = evaluate(args.engine, args.output_dir, args.prices)
    print(json.dumps({key: result[key] for key in ("passed", "checks_passed", "checks_total")}, indent=2))
    sys.exit(0 if result["passed"] else 1)
