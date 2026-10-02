"""Experiment evaluation: synthetic surcharge, and a refusal on the switchback log."""

from scripts.evaluate_experiment import evaluate_experiment
from tests.test_dpe_connector import create_dummy_dpe_db


def test_evaluate_experiment_synthetic():
    res = evaluate_experiment(source="synthetic")
    assert isinstance(res, dict)
    assert res["n_c"] > 0
    assert res["n_t"] > 0
    assert isinstance(res["cuped_delta"], float)
    assert isinstance(res["p_value"], float)
    assert "Experiment Evaluation Report" in res["markdown_report"]


def test_evaluate_experiment_dpe_is_not_a_row_trial(tmp_path):
    db_path = tmp_path / "eval_dpe.db"
    create_dummy_dpe_db(db_path, n_rows=40)

    res = evaluate_experiment(source="dpe", dpe_db_path=db_path)
    assert res["identified_ite"] is False
    assert "p_value" not in res
    assert "virtual hour" in res["markdown_report"].lower()
