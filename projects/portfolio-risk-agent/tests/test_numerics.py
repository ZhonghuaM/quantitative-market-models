"""Numerical regression cases and executable independent reference comparisons."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate import evaluate  # noqa: E402
from reference import historical_tail, kupiec_reference  # noqa: E402


def test_fractional_tail_mass_is_not_mean_above_interpolated_var():
    # Four equally likely losses. The upper 37.5% tail contains all of 4
    # and half of 3, so ES is 11/3, while the threshold exceedance mean is 3.5.
    var, es = historical_tail(np.array([-1.0, -2.0, -3.0, -4.0]), 0.625)
    assert var == pytest.approx(2.875)
    assert es == pytest.approx(11 / 3)
    assert es != pytest.approx(3.5)


def test_all_positive_returns_keep_negative_loss_quantiles():
    var, es = historical_tail(np.array([0.01, 0.02, 0.03, 0.04]), 0.625)
    assert var == pytest.approx(-0.02125)
    assert es == pytest.approx(-0.02 / 1.5)


def test_kupiec_zero_breaches_do_not_automatically_pass():
    result = kupiec_reference(0, 1000, 0.01)
    assert result["lr_statistic"] == pytest.approx(20.1006717070029)
    assert result["p_value"] == pytest.approx(7.34708677007e-6)
    assert kupiec_reference(1000, 1000, 0.01)["p_value"] < 1e-100


def test_native_engine_against_independent_references(tmp_path):
    configured = os.environ.get("QFA_ENGINE")
    engine = Path(configured) if configured else ROOT / "build/qfa_engine"
    if not engine.is_file():
        if configured:
            pytest.fail("QFA_ENGINE was supplied but does not name an executable file")
        pytest.skip("Build qfa_engine or set QFA_ENGINE to run native integration checks")
    report = evaluate(engine, tmp_path / "evaluation")
    failed = [row for row in report["checks"] if not row["passed"]]
    assert not failed, failed
    assert report["passed"]
