"""Small, strict tool surface shared by the model and local dispatcher.

Tool arguments contain identifiers and bounded analytical parameters, never
commands, credentials, URLs or filesystem paths. Schema validation is local;
server-side strict mode is an additional constraint, not a trust boundary.
"""

from __future__ import annotations

import copy
import json
import math

from jsonschema import Draft202012Validator, FormatChecker


class ToolArgumentsError(ValueError):
    """A public, bounded error that is safe to return to the model."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_DATASET = {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,80}$"}
_WEIGHTS = {
    "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
    "minItems": 1, "maxItems": 8,
}
_CONFIDENCE = {"type": "number", "exclusiveMinimum": 0.5, "maximum": 0.999}
_DATE = {"type": "string", "format": "date", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": copy.deepcopy(properties),
            "required": list(properties), "additionalProperties": False}


TOOL_PARAMETERS = {
    "load_market_data": _object({
        "source": {"type": "string", "enum": ["fixture", "yahoo"]},
        "tickers": {"type": "array", "items": {
            "type": "string", "pattern": "^[A-Z^][A-Z0-9_.^=-]{0,19}$"},
            "minItems": 1, "maxItems": 8, "uniqueItems": True},
        "start": _DATE, "end": _DATE,
    }),
    "estimate_risk": _object({
        "dataset_id": _DATASET, "weights": _WEIGHTS, "confidence": _CONFIDENCE,
    }),
    "simulate_risk": _object({
        "dataset_id": _DATASET, "weights": _WEIGHTS, "confidence": _CONFIDENCE,
        "paths": {"type": "integer", "minimum": 100, "maximum": 1_000_000},
        "horizon": {"type": "integer", "minimum": 1, "maximum": 20},
        "seed": {"type": "integer", "minimum": 0, "maximum": 4_294_967_295},
    }),
    "optimise_portfolio": _object({
        "dataset_id": _DATASET, "weights": _WEIGHTS, "confidence": _CONFIDENCE,
        "train_observations": {"type": "integer", "minimum": 20, "maximum": 100_000},
        "frontier_points": {"type": "integer", "minimum": 2, "maximum": 100},
    }),
    "backtest_var": _object({
        "dataset_id": _DATASET, "weights": _WEIGHTS, "confidence": _CONFIDENCE,
        "window": {"type": "integer", "minimum": 20, "maximum": 100_000},
    }),
    "write_report": _object({"dataset_id": _DATASET}),
}

_DESCRIPTIONS = {
    "load_market_data": "Load the requested fixture or Yahoo adjusted closes; dates are start-inclusive and end-exclusive. Return a dataset identifier and provenance.",
    "estimate_risk": "Compute historical portfolio risk from the identified dataset and requested long-only weights.",
    "simulate_risk": "Simulate portfolio risk with the requested fixed seed, path count and horizon.",
    "optimise_portfolio": "Fit long-only allocations using the initial training observations, then evaluate on subsequent observations.",
    "backtest_var": "Backtest one-observation-ahead VaR with an exclusively past rolling window.",
    "write_report": "Write an evidence-linked report from persisted results after all requested analyses succeed. No arbitrary file paths are accepted.",
}

TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": _DESCRIPTIONS[name],
     "parameters": copy.deepcopy(parameters), "strict": True}
    for name, parameters in TOOL_PARAMETERS.items()
]


def validate_tool_arguments(name: str, arguments: dict) -> None:
    """Reject unknown tools, extra fields and invalid/oversized JSON arguments."""
    if name not in TOOL_PARAMETERS:
        raise ToolArgumentsError("unknown_tool", "This tool is not in the allowed tool set.")
    try:
        encoded = json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError):
        raise ToolArgumentsError("invalid_arguments", "Arguments must be finite JSON values.") from None
    if len(encoded) > 16_384:
        raise ToolArgumentsError("invalid_arguments", "Arguments exceed the size limit.")
    validator = Draft202012Validator(TOOL_PARAMETERS[name], format_checker=FormatChecker())
    error = next(validator.iter_errors(arguments), None)
    if error is not None:
        raise ToolArgumentsError("invalid_arguments", f"Arguments do not match the tool schema ({error.validator}).")
    if "weights" in arguments and not math.isclose(sum(arguments["weights"]), 1.0, abs_tol=1e-8):
        raise ToolArgumentsError("invalid_arguments", "Long-only weights must sum to one.")
    if name == "load_market_data" and arguments["start"] >= arguments["end"]:
        raise ToolArgumentsError("invalid_arguments", "The exclusive end date must be after the start date.")
