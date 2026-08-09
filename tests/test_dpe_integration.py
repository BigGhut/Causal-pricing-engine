"""Integration tests for DPE API and CPE API interoperability."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app as cpe_app, apply_model_payload
from src.causal.uplift_models import TLearner
from sklearn.linear_model import LogisticRegression
import numpy as np


@pytest.fixture
def dummy_cpe_client():
    # Setup dummy model in CPE state
    X = np.random.randn(20, 7)
    y = np.random.binomial(1, 0.5, 20)
    t = np.random.binomial(1, 0.5, 20)

    model = TLearner(base_estimator=LogisticRegression())
    model.fit(X, y, t)

    payload = {
        "model": model,
        "model_name": "t_learner_test",
        "feature_columns": [
            "distance_km",
            "duration_sec",
            "price",
            "surge_bonus",
            "hour_of_day",
            "past_trips",
            "avg_surge",
        ],
        "uplift_threshold": 0.05,
    }
    apply_model_payload(payload)

    client = TestClient(cpe_app)
    return client


def test_cpe_predict_uplift_dpe_payload(dummy_cpe_client):
    payload = {
        "user_id": "search_12345",
        "features": {
            "distance_km": 12.5,
            "duration_sec": 650.0,
            "price": 350.0,
            "surge_bonus": 50.0,
            "hour_of_day": 14.0,
            "past_trips": 5.0,
            "avg_surge": 20.0,
        },
    }

    response = dummy_cpe_client.post("/predict_uplift", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == "search_12345"
    assert "uplift_score" in data
    assert "recommended_treatment" in data
    assert "optimal_discount_pct" in data
    assert data["model_name"] == "t_learner_test"
