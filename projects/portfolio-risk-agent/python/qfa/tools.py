"""Stateful, configuration-bound tool dispatcher; numerical work stays native."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .market_data import load_market_data
from .native import NativeEngine
from .schemas import validate_tool_arguments

PROJECT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj: object) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def canonical_calls(config: dict) -> list[dict]:
    """Explicit offline demonstration, never represented as an LLM-selected plan."""
    calls = [
        {
            "name": "load_market_data",
            "arguments": {k: config[k] for k in ("source", "tickers", "start", "end")},
        }
    ]
    common = {
        "dataset_id": "$dataset_id",
        "weights": config["weights"],
        "confidence": config["confidence"],
    }
    for name, extra in (
        ("estimate_risk", ()),
        ("simulate_risk", ("paths", "horizon", "seed")),
        ("optimise_portfolio", ("train_observations", "frontier_points")),
        ("backtest_var", ("window",)),
    ):
        calls.append({"name": name, "arguments": {**common, **{k: config[k] for k in extra}}})
    calls.append({"name": "write_report", "arguments": {"dataset_id": "$dataset_id"}})
    return calls


class ToolSession:
    def __init__(self, config: dict, output_dir: Path, engine: Path, mode: str):
        self.config = json.loads(json.dumps(config, allow_nan=False))
        for call in canonical_calls(self.config):
            validate_tool_arguments(
                call["name"],
                {k: "pending" if v == "$dataset_id" else v for k, v in call["arguments"].items()},
            )
        if abs(sum(config["weights"]) - 1.0) > 1e-8 or len(config["weights"]) != len(
            config["tickers"]
        ):
            raise ValueError("Weights must match ticker order and sum to one")
        if not 0 < config.get("portfolio_value", 1_000_000) < 1e15:
            raise ValueError("portfolio_value must be positive and finite")
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise ValueError("Choose an empty output directory; existing runs are immutable")
        self.engine = NativeEngine(engine)
        self.mode = mode
        self.dataset = None
        self.results: dict[str, dict] = {}
        self.audit: list[dict] = []
        save_json(self.output_dir / "input_config.json", self.config)

    def execute(self, name: str, args: dict) -> dict:
        started = time.perf_counter()
        event = {
            "sequence": len(self.audit),
            "tool": name,
            "arguments": json.loads(json.dumps(args, allow_nan=False)),
        }
        try:
            validate_tool_arguments(name, args)
            result = self._dispatch(name, args)
            event.update(status="success", evidence=result.get("evidence_id"))
            return result
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started
            self.audit.append(event)
            save_json(self.output_dir / "tool_audit.json", self.audit)

    def _lock_arguments(self, args: dict) -> None:
        for key, value in args.items():
            if key in self.config and value != self.config[key]:
                raise ValueError(
                    f"{key} differs from the submitted configuration; start a new run to change it"
                )

    def _dispatch(self, name: str, args: dict) -> dict:
        self._lock_arguments(args)
        if name == "load_market_data":
            if self.dataset is not None:
                return self.dataset.summary()
            self.dataset = load_market_data(
                **args,
                output_dir=self.output_dir / "data",
                fixture_path=PROJECT / "data" / "synthetic_prices.csv",
            )
            return self.dataset.summary()
        if self.dataset is None or args["dataset_id"] != self.dataset.dataset_id:
            raise ValueError("Load the configured dataset first and use its returned dataset_id")
        if name == "write_report":
            required = {"estimate_risk", "simulate_risk", "optimise_portfolio", "backtest_var"}
            if required - self.results.keys():
                raise ValueError(
                    "Report requires risk, simulation, optimisation and coverage backtest evidence"
                )
            from .report import write_report

            files = write_report(
                self.output_dir, self.results, self.dataset.metadata, self.config, self.mode
            )
            return {"evidence_id": "report", "files": files, "dataset_id": self.dataset.dataset_id}
        mapping = {
            "estimate_risk": "risk",
            "simulate_risk": "monte-carlo",
            "optimise_portfolio": "optimise",
            "backtest_var": "backtest",
        }
        if name not in mapping:
            raise ValueError("Unknown tool")
        params = {
            k.replace("train_observations", "train").replace(
                "frontier_points", "frontier-points"
            ): v
            for k, v in args.items()
            if k != "dataset_id"
        }
        result = self.engine.run(mapping[name], self.dataset.prices_path, **params)
        if result.get("assets") != list(self.config["tickers"]):
            raise ValueError("Native engine asset order differs from the submitted portfolio")
        self.results[name] = result
        save_json(self.output_dir / f"{name}.json", result)
        # Large forecast arrays remain in the evidence file, not in model context.
        summary = {k: v for k, v in result.items() if k != "forecasts"}
        summary.update(
            evidence_id=name, dataset_id=self.dataset.dataset_id, artifact=f"{name}.json"
        )
        if "forecasts" in result:
            summary["forecast_count"] = len(result["forecasts"])
        return summary

    def finish(self, agent_result: dict) -> None:
        save_json(self.output_dir / "agent_run.json", agent_result)
        versions = {}
        for package in (
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "reportlab",
            "openai",
            "yfinance",
            "jsonschema",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None

        def git(*args):
            try:
                return subprocess.check_output(
                    ["git", *args], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None

        sources = [
            p
            for folder in ("native", "python", "evaluation", "tests", "configs")
            for p in (PROJECT / folder).rglob("*")
            if p.is_file() and p.suffix in {".cpp", ".c", ".hpp", ".h", ".py", ".R", ".json"}
        ]
        sources += [
            PROJECT / n
            for n in (
                "CMakeLists.txt",
                "pyproject.toml",
                "requirements-lock.txt",
                "data/generate_synthetic.py",
            )
            if (PROJECT / n).exists()
        ]
        status = git("status", "--porcelain")
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "status": agent_result["status"],
            "model": agent_result.get("model"),
            "configuration": self.config,
            "source_commit": git("rev-parse", "HEAD"),
            "source_dirty": None if status is None else bool(status),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": versions,
            "native_binary_sha256": sha256(self.engine.executable),
            "native_build": json.loads(
                subprocess.check_output(
                    [str(self.engine.executable), "version"], text=True, timeout=10
                )
            ),
            "source_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(sources)},
            "outputs_sha256": {
                str(p.relative_to(self.output_dir)): sha256(p)
                for p in sorted(self.output_dir.rglob("*"))
                if p.is_file()
            },
            "reproducibility_boundary": "Numerical seed and archived data can reproduce calculations within tolerance; LLM choices/text and external data are not deterministic.",
        }
        save_json(self.output_dir / "run_manifest.json", manifest)
