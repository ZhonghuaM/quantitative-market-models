"""User-initiated live evaluation of the public full-analysis tool contract.

Partial-report intents in test_cases.json are intentionally not scored: the
current application requires four analyses before writing its evidence report.
This harness makes real API calls only when explicitly invoked with a model
and an OPENAI_API_KEY. It never substitutes a scripted run for live evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Support direct invocation without requiring an editable package install.
from qfa.agent import run_openai  # noqa: E402
from qfa.schemas import TOOL_SCHEMAS  # noqa: E402
from qfa.tools import ToolSession, canonical_calls, save_json  # noqa: E402


def score_full_analysis(run: dict, config: dict) -> dict:
    """Score exact observable calls, separately from unreviewed model prose.

    Strict success requires one successful call to each expected tool, load
    first and report last, exact configured arguments and application success.
    The four independent analysis tools may be called in any order. Recovery
    can complete a run while failing this stricter one-attempt contract.
    """
    events = [event for event in run["events"] if event.get("type") == "tool_call"]
    expected = {call["name"]: call["arguments"] for call in canonical_calls(config)}
    dataset_id = None
    for event in events:
        if event["name"] == "load_market_data" and event["success"]:
            dataset_id = event["output"]["result"].get("dataset_id")
            break
    checks = []
    for event in events:
        target = expected.get(event["name"])
        if target is not None:
            target = {key: dataset_id if value == "$dataset_id" else value
                      for key, value in target.items()}
        checks.append({"call_id": event["call_id"], "tool": event["name"],
                       "allowed_expected_tool": target is not None,
                       "arguments_match_configured_contract": target is not None and event["arguments"] == target,
                       "tool_success": bool(event["success"])})
    names = [event["name"] for event in events]
    exact_tools = len(names) == len(expected) and set(names) == set(expected)
    boundaries = bool(names) and names[0] == "load_market_data" and names[-1] == "write_report"
    completed = run["status"] == "completed" and run["report_written"]
    api_evidence = run.get("mode") == "openai" and bool(run.get("responses"))
    all_arguments = bool(checks) and all(row["arguments_match_configured_contract"] for row in checks)
    all_succeeded = bool(checks) and all(row["tool_success"] for row in checks)
    return {
        "strict_success": bool(api_evidence and completed and exact_tools and boundaries and all_arguments and all_succeeded),
        "openai_response_evidence_present": api_evidence,
        "application_completed": bool(completed), "application_status": run["status"],
        "each_expected_tool_exactly_once": exact_tools,
        "load_first_report_last": boundaries,
        "all_arguments_match": all_arguments, "all_tool_calls_succeeded": all_succeeded,
        "observed_tool_names": names, "expected_tool_names": list(expected), "call_checks": checks,
        "unsupported_claim_review": "not_performed; final model prose requires separate human review",
        "interpretation": "One public full-analysis case; this is not held-out LLM accuracy or a financial-performance evaluation.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An explicit API model identifier you can access")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="An empty directory inside this project's ignored runs/ or system temporary storage")
    parser.add_argument("--engine", type=Path, default=ROOT / "build" / "qfa_engine")
    parser.add_argument("--max-calls", type=int, default=12)
    args = parser.parse_args(argv)
    if not args.model.strip() or not os.environ.get("OPENAI_API_KEY"):
        parser.error("Live evaluation requires --model and OPENAI_API_KEY; no scripted scores are substituted")
    if args.max_calls < 1:
        parser.error("--max-calls must be positive")
    output = args.output_dir.resolve()
    allowed_roots = [(ROOT / "runs").resolve(), Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve()]
    if not any(output.is_relative_to(root) and output != root for root in allowed_roots):
        parser.error("Save live artifacts under ignored runs/ or a new temporary directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose an empty output directory; existing evaluations are not overwritten")
    if not args.engine.is_file():
        parser.error("Build the native engine before running the live evaluation")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install the optional OpenAI dependency before running live evaluation")
    suite_path = ROOT / "evaluation" / "test_cases.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    case = next(item for item in suite["cases"] if item["id"] == "full_scripted_contract")
    config = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
    # The public source case was also designed for a scripted contract test.
    # Record this exact adaptation instead of pretending it is held-out text.
    prompt = (
        "Perform the full portfolio analysis through the provided tools: load the configured "
        "synthetic data, estimate historical risk, simulate risk, optimise the allocation with "
        "a later holdout, backtest VaR, and write the evidence report. This run uses actual "
        "language-model tool selection; do not describe it as a scripted execution. "
        "Keep all submitted parameters unchanged. State limitations supported by tool evidence.\n"
        "Submitted configuration:\n" + json.dumps(config, sort_keys=True)
    )
    try:
        output.mkdir(parents=True, exist_ok=True)
        session = ToolSession(config, output / "run", args.engine, "openai")
        specification = {
            "case_id": "full_analysis", "source_case_id": case["id"],
            "source_case_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            "source_prompt": case["prompt"], "effective_prompt": prompt,
            "prompt_adaptation": "The full scripted contract is explicitly requested as a live tool-selection run.",
            "expected_tool_names": case["expected_tool_names"],
            "expected_arguments": canonical_calls(config), "configuration": config,
            "offered_tools": TOOL_SCHEMAS, "requested_model": args.model,
            "excluded_source_cases": [item["id"] for item in suite["cases"] if item["id"] != case["id"]],
            "scope": "Public full-report contract only; subset and refusal cases need separate completion policies before live scoring.",
        }
        save_json(output / "evaluation_specification.json", specification)
        run = run_openai(client=OpenAI(timeout=30, max_retries=1), model=args.model,
                         prompt=prompt, execute=session.execute, max_calls=args.max_calls)
        session.finish(run)
        evaluation = {
            "schema_version": 1, "mode": "live_openai_evaluation", "case_id": "full_analysis",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_model": args.model, "responses": run["responses"], "usage": run["usage"],
            "scores": score_full_analysis(run, config),
        }
        save_json(output / "live_evaluation.json", evaluation)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Live evaluation could not complete ({type(exc).__name__}). No success score was recorded.", file=sys.stderr)
        return 2
    print(json.dumps({"mode": evaluation["mode"], "case_id": evaluation["case_id"],
                      "strict_success": evaluation["scores"]["strict_success"],
                      "application_status": run["status"], "output_dir": str(output)}))
    return 0 if evaluation["scores"]["strict_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
