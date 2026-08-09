"""Tests for the experiment evaluation pipeline."""

import pytest
from src.data.dpe_connector import DEFAULT_DPE_DB_PATH
from scripts.evaluate_experiment import evaluate_experiment


def test_evaluate_experiment_synthetic():
    res = evaluate_experiment(source="synthetic")
    assert isinstance(res, dict)
    assert res["n_c"] > 0
    assert res["n_t"] > 0
    assert isinstance(res["cuped_delta"], float)
    assert isinstance(res["p_value"], float)
    assert "markdown_report" in res
    assert "Experiment Evaluation Report" in res["markdown_report"]


def test_evaluate_experiment_dpe_real_db():
    if not DEFAULT_DPE_DB_PATH.exists():
        pytest.skip(f"DPE DB not found at {DEFAULT_DPE_DB_PATH}")

    res = evaluate_experiment(source="dpe")
    assert isinstance(res, dict)
    assert res["n_c"] > 0
    assert res["n_t"] > 0
    assert 0.0 <= res["p_value"] <= 1.0
    assert "Verdict" in res["markdown_report"]
