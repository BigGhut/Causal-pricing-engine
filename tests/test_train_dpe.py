"""DPE logs are not a training set for the surcharge model."""

from pathlib import Path

from scripts.train import train_and_select
from tests.test_dpe_connector import create_dummy_dpe_db


def test_train_refuses_dpe_switchback(tmp_path: Path):
    db_path = tmp_path / "train_dpe_fixture.db"
    create_dummy_dpe_db(db_path, n_rows=60)

    result = train_and_select(source="dpe", dpe_db_path=db_path, random_state=0)

    assert result["refused"] is True
    assert result["identified_ite"] is False
    assert "model_path" not in result
    assert result["design"]["arm_clock_recorded"] is False
    assert "driver" in result["design"]["report"].lower()
