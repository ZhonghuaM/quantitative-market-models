"""Typed, shell-free process boundary to the numerical engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class NativeError(RuntimeError):
    """A native calculation failed or returned an invalid payload."""


class NativeEngine:
    COMMANDS = {"risk", "optimise", "monte-carlo", "backtest"}
    OPTIONS = {
        "weights",
        "confidence",
        "train",
        "frontier-points",
        "paths",
        "horizon",
        "seed",
        "window",
    }

    def __init__(self, executable: Path, timeout: float = 90):
        self.executable = executable.resolve()
        self.timeout = timeout
        if not self.executable.is_file():
            raise NativeError(
                "Build the native engine first: cmake -S . -B build && cmake --build build"
            )

    def run(self, command: str, prices: Path, **parameters) -> dict:
        if command not in self.COMMANDS or set(parameters) - self.OPTIONS:
            raise NativeError("Unknown native command or parameter")
        args = [str(self.executable), command, "--prices", str(prices.resolve())]
        for name, value in parameters.items():
            if isinstance(value, (list, tuple)):
                value = ",".join(format(float(v), ".17g") for v in value)
            args.extend([f"--{name}", str(value)])
        try:
            completed = subprocess.run(
                args, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            result = json.loads(
                completed.stdout, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))
            )
        except subprocess.TimeoutExpired as exc:
            raise NativeError("Native calculation exceeded its time limit") from exc
        except (OSError, ValueError) as exc:
            raise NativeError("Native engine failed to return valid finite JSON") from exc
        if not isinstance(result, dict):
            raise NativeError("Native output must be an object")
        if completed.returncode or "error" in result:
            error = result.get("error", {}) if isinstance(result, dict) else {}
            raise NativeError(str(error.get("message", "Native calculation failed"))[:300])
        return result
