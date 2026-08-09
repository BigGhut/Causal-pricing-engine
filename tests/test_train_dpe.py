"""Tests for training pipeline using DPE data source with SQLite fixture."""

from pathlib import Path
import joblib
import pytest

from scripts.train import train_and_select
from src.data.dpe_connector import DPE_FEATURE_COLUMNS
from tests.test_dpe_connector import create_dummy_dpe_db


def test_train_and_select_dpe_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify train_and_select with source='dpe' produces valid artifact with DPE_FEATURE_COLUMNS."""
    db_path = tmp_path / "train_dpe_fixture.db"
    create_dummy_dpe_db(db_path, n_rows=60)

    # Run training pipeline with dpe_db_path override
    result = train_and_select(source="dpe", dpe_db_path=db_path, random_state=0)

    assert "best_name" in result
    assert "model_path" in result
    assert Path(result["model_path"]).exists()

    # Inspect dumped joblib artifact
    artifact = joblib.load(result["model_path"])
    assert artifact["source"] == "dpe"
    assert artifact["feature_columns"] == DPE_FEATURE_COLUMNS
    assert "model" in artifact
    assert "metrics" in artifact

    # Test prediction with artifact model on sample DPE features
    model = artifact["model"]
    sample_features = [[5.0, 600.0, 20.0, 1.5, 14.0, 3.0, 1.1]]
    if hasattr(model, "predict_uplift"):
        preds = model.predict_uplift(sample_features)
    else:
        preds = model.effect(sample_features)
    assert len(preds) == 1
