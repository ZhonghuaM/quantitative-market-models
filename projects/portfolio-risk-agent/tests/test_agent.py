"""Contract tests use mock Responses objects; these are not live-model evals."""

import copy
import json
from types import SimpleNamespace

import pytest

from qfa.agent import run_openai, run_scripted
from qfa.schemas import TOOL_SCHEMAS, ToolArgumentsError, validate_tool_arguments


LOAD = {"source": "fixture", "tickers": ["SYNTH_A", "SYNTH_B"],
        "start": "2020-01-01", "end": "2023-01-01"}
RISK = {"dataset_id": "data_123", "weights": [0.6, 0.4], "confidence": 0.95}


def call(name, arguments, call_id="call-1"):
    return {"type": "function_call", "name": name, "call_id": call_id,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}


def response(output, text="", number=1):
    return {"id": f"resp-{number}", "model": "mock-model-snapshot", "status": "completed",
            "_request_id": f"req-{number}", "output": output, "output_text": text,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


class MockResponses:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.handler(kwargs, len(self.requests))


def test_dynamic_responses_loop_preserves_reasoning_and_uses_tool_result():
    reasoning = {"type": "reasoning", "id": "reasoning-1", "summary": [],
                 "encrypted_content": "opaque-state"}
    executed = []

    def execute(name, args):
        executed.append((name, args))
        if name == "load_market_data":
            return {"dataset_id": "data_123"}
        if name == "estimate_risk":
            assert args["dataset_id"] == "data_123"
            return {"historical_var": 0.02}
        return {"report": "report.pdf"}

    def handler(request, turn):
        assert request["model"] == "user-chosen-model"
        assert request["store"] is False
        assert request["parallel_tool_calls"] is False
        assert request["include"] == ["reasoning.encrypted_content"]
        if turn == 1:
            return response([reasoning, call("load_market_data", LOAD)], number=turn)
        assert request["input"][1] == reasoning
        last = json.loads(request["input"][-1]["output"])
        if turn == 2:
            args = {**RISK, "dataset_id": last["result"]["dataset_id"]}
            return response([call("estimate_risk", args, "call-2")], number=turn)
        if turn == 3:
            assert last["result"]["historical_var"] == 0.02
            return response([call("write_report", {"dataset_id": "data_123"}, "call-3")], number=turn)
        return response([{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Report written from recorded evidence."}]}], number=turn)

    mocked = MockResponses(handler)
    run = run_openai(client=SimpleNamespace(responses=mocked), model="user-chosen-model",
                     prompt="Use the requested fixture.", execute=execute)
    assert run["status"] == "completed"
    assert run["report_written"] is True
    assert run["final_text"] == "Report written from recorded evidence."
    assert [name for name, _ in executed] == ["load_market_data", "estimate_risk", "write_report"]
    assert run["usage"] == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    assert run["responses"][0]["request_id"] == "req-1"
    assert run["responses"][0]["model"] == "mock-model-snapshot"
    assert run["events"][0]["type"] == "tool_call"
    json.dumps(run, allow_nan=False)


@pytest.mark.parametrize("name,args,code", [
    ("run_shell", {"command": "echo unsafe"}, "unknown_tool"),
    ("write_report", {"dataset_id": "data_123", "path": "/tmp/unapproved"}, "invalid_arguments"),
    ("estimate_risk", {**RISK, "weights": [1.0, 1.0]}, "invalid_arguments"),
    ("write_report", '{"dataset_id":', "invalid_arguments"),
    ("estimate_risk", '{"weights": [NaN]}', "invalid_arguments"),
])
def test_bad_tool_call_is_rejected_locally_and_error_is_sent_back(name, args, code):
    executed = []

    def handler(request, turn):
        if turn == 1:
            return response([call(name, args)])
        error = json.loads(request["input"][-1]["output"])
        assert error["ok"] is False
        assert error["error"]["code"] == code
        assert len(error["error"]["message"]) < 200
        return response([], "Unable to complete.")

    mocked = MockResponses(handler)
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda name, args: executed.append((name, args)))
    assert not executed
    assert run["status"] == "missing_report"
    assert run["final_text"] == ""


def test_tool_exception_details_are_not_sent_to_model():
    def execute(name, args):
        raise RuntimeError("PRIVATE_API_TOKEN=secret /private/person/report.csv")

    mocked = MockResponses(lambda request, turn: response([call("estimate_risk", RISK)]))
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=execute, max_rounds=1)
    assert run["status"] == "max_rounds"
    assert "PRIVATE_API_TOKEN" not in json.dumps(run)
    assert run["events"][0]["output"]["error"]["code"] == "tool_failed"


def test_final_answer_without_successful_report_is_not_accepted():
    mocked = MockResponses(lambda request, turn: response([], "I wrote a report."))
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda *_: {})
    assert run["status"] == "missing_report"
    assert run["final_text"] == ""
    assert run["report_written"] is False


@pytest.mark.parametrize("limit_name,limit,status,expected_calls", [
    ("max_rounds", 2, "max_rounds", 2), ("max_calls", 1, "max_calls", 1),
])
def test_repeated_tool_requests_are_bounded(limit_name, limit, status, expected_calls):
    mocked = MockResponses(lambda request, turn: response([call("estimate_risk", RISK, f"call-{turn}")]))
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda *_: {"var": 0.02}, **{limit_name: limit})
    assert run["status"] == status
    assert run["tool_calls"] == expected_calls


def test_incomplete_response_and_api_failure_never_execute_tools():
    partial = {**response([call("estimate_risk", RISK)]), "status": "incomplete"}
    mocked = MockResponses(lambda *_: partial)
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda *_: pytest.fail("Incomplete call must not execute"))
    assert run["status"] == "response_incomplete"

    def fail(*_):
        raise RuntimeError("secret credential")
    mocked = MockResponses(fail)
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda *_: pytest.fail("API failure must not execute"))
    assert run["status"] == "api_error"
    assert "secret credential" not in json.dumps(run)


def test_scripted_workflow_is_explicit_and_substitutes_only_dataset_identifier():
    executed = []

    def execute(name, args):
        executed.append((name, args))
        return {"dataset_id": "data_123"} if name == "load_market_data" else {"report": "report.pdf"}
    run = run_scripted(calls=[{"name": "load_market_data", "arguments": LOAD},
                              {"name": "write_report", "arguments": {"dataset_id": "$dataset_id"}}],
                       execute=execute)
    assert run["mode"] == "scripted_workflow"
    assert run["model"] is None
    assert run["status"] == "completed"
    assert "No language model was called" in run["final_text"]
    assert executed[1][1] == {"dataset_id": "data_123"}
    assert run["responses"] == []


def test_strict_schemas_reject_extra_arguments_and_native_limit_excess():
    for tool in TOOL_SCHEMAS:
        assert tool["strict"] is True
        assert tool["parameters"]["additionalProperties"] is False
        assert set(tool["parameters"]["required"]) == set(tool["parameters"]["properties"])
    with pytest.raises(ToolArgumentsError):
        validate_tool_arguments("write_report", {"dataset_id": "data_123", "path": "report.pdf"})
    with pytest.raises(ToolArgumentsError):
        validate_tool_arguments("simulate_risk", {**RISK, "paths": 1000, "horizon": 21, "seed": 42})


def test_duplicate_call_identifier_cannot_repeat_a_tool_side_effect():
    executed = []
    mocked = MockResponses(lambda *_: response([call("estimate_risk", RISK, "same-call")]))
    run = run_openai(client=SimpleNamespace(responses=mocked), model="mock", prompt="Test",
                     execute=lambda name, args: executed.append(name) or {"var": 0.02})
    assert run["status"] == "invalid_response"
    assert executed == ["estimate_risk"]
