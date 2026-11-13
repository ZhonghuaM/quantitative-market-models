"""Essential native-backed workflow checks; all data are local and synthetic."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qfa.agent import run_scripted
from qfa.schemas import ToolArgumentsError
from qfa.tools import PROJECT, ToolSession, canonical_calls


@pytest.fixture(scope="module")
def engine():
    configured = os.environ.get("QFA_ENGINE")
    path = Path(configured) if configured else PROJECT / "build" / "qfa_engine"
    if not path.is_file():
        if configured:
            pytest.fail("QFA_ENGINE does not name an executable file")
        pytest.skip("Build qfa_engine to exercise native integration")
    return path


@pytest.fixture
def config():
    return json.loads((PROJECT / "configs" / "demo.json").read_text())


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, engine):
    config = json.loads((PROJECT / "configs" / "demo.json").read_text())
    output = tmp_path_factory.mktemp("native-workflow") / "run"
    session = ToolSession(config, output, engine, "scripted_workflow")
    result = run_scripted(calls=canonical_calls(config), execute=session.execute)
    assert result["status"] == "completed", result
    session.finish(result)
    return output, session, result


def test_complete_workflow_keeps_evidence_report_and_manifest_consistent(completed_run):
    output, session, result = completed_run
    assert result["mode"] == "scripted_workflow"
    assert result["tool_calls"] == 6
    assert result["report_written"] is True
    assert result["responses"] == []
    assert result["model"] is None
    assert all(event["success"] for event in result["events"])
    evidence_ids = {event["output"]["result"].get("evidence_id") for event in result["events"]}
    assert {"estimate_risk", "simulate_risk", "optimise_portfolio", "backtest_var", "report"} <= evidence_ids
    assert (output / "portfolio_report.pdf").read_bytes().startswith(b"%PDF-")
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["configuration"] == session.config
    for filename, expected in manifest["outputs_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    audit = json.loads((output / "tool_audit.json").read_text())
    assert [row["tool"] for row in audit] == [call["name"] for call in canonical_calls(session.config)]
    assert all(row["status"] == "success" for row in audit)


def test_report_numbers_are_derived_from_persisted_native_evidence(completed_run):
    output, _, _ = completed_run
    risk = json.loads((output / "estimate_risk.json").read_text())
    simulation = json.loads((output / "simulate_risk.json").read_text())
    optimum = json.loads((output / "optimise_portfolio.json").read_text())
    backtest = json.loads((output / "backtest_var.json").read_text())
    estimates = pd.read_csv(output / "risk_estimates.csv").set_index("method")
    assert estimates.loc["historical", "var_fraction"] == pytest.approx(risk["historical_var"])
    assert estimates.loc["historical", "es_fraction"] == pytest.approx(risk["historical_es"])
    assert estimates.loc["lognormal_monte_carlo", "var_fraction"] == pytest.approx(simulation["monte_carlo_var"])
    allocation = pd.read_csv(output / "portfolio_allocation.csv")
    assert allocation["asset"].tolist() == risk["assets"]
    np.testing.assert_allclose(allocation["minimum_variance_weight"], optimum["minimum_variance"]["weights"])
    recorded = pd.read_csv(output / "var_backtest.csv")
    np.testing.assert_allclose(recorded["var"], [row["var"] for row in backtest["forecasts"]])
    assert recorded["date"].tolist() == [row["date"] for row in backtest["forecasts"]]
    text = (output / "portfolio_report.md").read_text()
    assert f"Historical VaR: {100 * risk['historical_var']:.2f}%" in text
    assert f"ES: {100 * risk['historical_es']:.2f}%" in text
    assert f"{100 * optimum['comparison']['minimum_variance']['holdout']['annualized_volatility']:.2f}%" in text


def test_existing_run_cannot_be_silently_overwritten(completed_run, engine):
    output, session, _ = completed_run
    original_manifest = (output / "run_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="empty output directory"):
        ToolSession(session.config, output, engine, "scripted_workflow")
    assert (output / "run_manifest.json").read_bytes() == original_manifest


def test_frozen_configuration_and_dataset_boundaries(tmp_path, config, engine):
    supplied = copy.deepcopy(config)
    session = ToolSession(supplied, tmp_path / "run", engine, "scripted_workflow")
    supplied["weights"][0] = 0.0
    assert session.config == config
    load = canonical_calls(config)[0]
    with pytest.raises(ValueError, match="differs from the submitted"):
        session.execute("load_market_data", {**load["arguments"], "tickers": ["SYNTH_MISSING"]})
    with pytest.raises(ValueError, match="Load the configured dataset"):
        session.execute("write_report", {"dataset_id": "unknown_dataset"})
    data = session.execute(load["name"], load["arguments"])
    risk = {"dataset_id": data["dataset_id"], "weights": config["weights"], "confidence": config["confidence"]}
    with pytest.raises(ValueError, match="differs from the submitted"):
        session.execute("estimate_risk", {**risk, "weights": [0.5, 0.3, 0.2]})
    with pytest.raises(ValueError, match="returned dataset_id"):
        session.execute("estimate_risk", {**risk, "dataset_id": "other_dataset"})
    with pytest.raises(ValueError, match="Report requires"):
        session.execute("write_report", {"dataset_id": data["dataset_id"]})
    with pytest.raises(ToolArgumentsError):
        session.execute("write_report", {"dataset_id": data["dataset_id"], "path": "/tmp/unapproved.pdf"})
    assert not session.results
    assert not (session.output_dir / "portfolio_report.pdf").exists()


def test_audit_captures_arguments_at_execution_time(tmp_path, config, engine):
    session = ToolSession(config, tmp_path / "run", engine, "scripted_workflow")
    arguments = canonical_calls(config)[0]["arguments"]
    session.execute("load_market_data", arguments)
    arguments["tickers"][0] = "MUTATED_AFTER_CALL"
    assert session.audit[0]["arguments"]["tickers"] == session.config["tickers"]


def _live_evaluator():
    spec = importlib.util.spec_from_file_location("qfa_live_evaluation_contract", PROJECT / "evaluation" / "live_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_evaluator_does_not_count_scripted_workflow_as_live_success(completed_run):
    _, session, result = completed_run
    evaluator = _live_evaluator()
    scored = evaluator.score_full_analysis(result, session.config)
    assert scored["each_expected_tool_exactly_once"] is True
    assert scored["application_completed"] is True
    assert scored["openai_response_evidence_present"] is False
    assert scored["strict_success"] is False


def test_live_evaluator_requires_credentials_before_creating_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "not-run"
    evaluator = _live_evaluator()
    with pytest.raises(SystemExit) as error:
        evaluator.main(["--model", "explicit-test-model", "--output-dir", str(output)])
    assert error.value.code == 2
    assert "no scripted scores are substituted" in capsys.readouterr().err
    assert not output.exists()
