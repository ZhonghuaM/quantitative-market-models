"""Two explicit modes: offline workflow and real OpenAI orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_openai, run_scripted
from .tools import PROJECT, ToolSession, canonical_calls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "openai"], default="demo")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "demo.json")
    parser.add_argument("--engine", type=Path, default=PROJECT / "build" / "qfa_engine")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", help="Explicit API model identifier; required for openai mode")
    parser.add_argument(
        "--request",
        default="Analyse my portfolio, estimate 99% VaR and propose a lower-estimated-variance allocation. Include Monte Carlo risk, holdout comparisons and a VaR backtest.",
    )
    parser.add_argument("--max-calls", type=int, default=12)
    args = parser.parse_args(argv)
    if args.mode == "openai" and (not args.model or not os.environ.get("OPENAI_API_KEY")):
        parser.error(
            "openai mode requires --model and OPENAI_API_KEY in the environment; demo makes no API calls"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = PROJECT / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    mode = "scripted_workflow" if args.mode == "demo" else "openai"
    try:
        session = ToolSession(config, args.output_dir, args.engine, mode)
        if args.mode == "openai":
            from openai import OpenAI

            client = OpenAI(timeout=30, max_retries=1)
            prompt = (
                args.request
                + "\nUse exactly this submitted configuration; do not replace it:\n"
                + json.dumps(config, sort_keys=True)
            )
            result = run_openai(
                client=client,
                model=args.model,
                prompt=prompt,
                execute=session.execute,
                max_calls=args.max_calls,
            )
        else:
            result = run_scripted(
                calls=canonical_calls(config), execute=session.execute, max_calls=args.max_calls
            )
        session.finish(result)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Run could not start: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "status": result["status"],
                "tool_calls": result["tool_calls"],
                "output_dir": str(session.output_dir),
                "report_written": result["report_written"],
            }
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
