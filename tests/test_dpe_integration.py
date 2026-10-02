"""CPE request contract used by DPE."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api.main import app as cpe_app, apply_model_payload
from src.causal.uplift_models import TLearner
from src.data.synthetic import FEATURE_COLUMNS


@pytest.fixture
def dummy_cpe_client():
    X = np.random.randn(20, len(FEATURE_COLUMNS))
    y = np.random.binomial(1, 0.5, 20)
    t = np.random.binomial(1, 0.5, 20)

    model = TLearner(base_estimator=LogisticRegression())
    model.fit(X, y, t)

    apply_model_payload(
        {
            "model": model,
            "model_name": "t_learner_test",
            "feature_columns": list(FEATURE_COLUMNS),
            "uplift_threshold": 0.05,
        }
    )
    return TestClient(cpe_app)


def test_cpe_predict_uplift_dpe_payload(dummy_cpe_client):
    payload = {
        "driver_id": "driver_7",
        "features": {
            "distance_km": 12.5,
            "duration_sec": 650.0,
            "hour_of_day": 14.0,
            "past_trips": 5.0,
            "avg_surge": 20.0,
        },
    }

    response = dummy_cpe_client.post("/predict_uplift", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["driver_id"] == "driver_7"
    assert "uplift_score" in data
    assert data["recommended_treatment"] in {"SURCHARGE", "KEEP_QUOTE", "NO_SURCHARGE"}
    assert "optimal_discount_pct" not in data
    assert "user_id" not in data
    assert data["model_name"] == "t_learner_test"


def test_post_treatment_fields_are_ignored(dummy_cpe_client):
    base = {
        "driver_id": "driver_7",
        "features": {
            "distance_km": 12.5,
            "duration_sec": 650.0,
            "hour_of_day": 14.0,
            "past_trips": 5.0,
            "avg_surge": 20.0,
        },
    }
    extra = {
        "driver_id": "driver_7",
        "features": {**base["features"], "price": 999.0, "surge_bonus": 80.0},
    }
    first = dummy_cpe_client.post("/predict_uplift", json=base).json()["uplift_score"]
    second = dummy_cpe_client.post("/predict_uplift", json=extra).json()["uplift_score"]
    assert first == pytest.approx(second)
