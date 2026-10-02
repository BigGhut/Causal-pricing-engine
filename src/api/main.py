"""FastAPI application for serving Causal ML Uplift decisions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.data.synthetic import FEATURE_COLUMNS
from src.evaluation.metrics import read_ranking_supports_decision

# Module-level model state populated at startup
_MODEL_STATE: dict[str, Any] = {
    "model": None,
    "feature_columns": FEATURE_COLUMNS,
    "model_name": None,
    "uplift_threshold": 0.05,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_model_path(configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    return _project_root() / path


def load_model_artifact(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load a joblib model payload and return its contents.

    Shared by the API lifespan and tests so both exercise the same path.
    """
    cfg = load_config()
    path = _resolve_model_path(str(model_path or cfg.api.model_path))
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid model artifact at {path}: expected dict with 'model'")
    return payload


def features_to_array(
    features: dict[str, float],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Convert a feature dict to a 2d array in the training column order."""
    cols = feature_columns or _MODEL_STATE["feature_columns"] or FEATURE_COLUMNS
    row = [float(features.get(c, 0.0)) for c in cols]
    return np.asarray([row], dtype=float)


def score_uplift(features: dict[str, float]) -> float:
    """Score a single feature dict with the loaded model.

    Returns:
        Predicted ITE (uplift) score.
    """
    model = _MODEL_STATE.get("model")
    if model is None:
        raise RuntimeError("Model is not loaded")

    X = features_to_array(features, _MODEL_STATE.get("feature_columns"))
    if hasattr(model, "predict_uplift"):
        scores = model.predict_uplift(X)
    elif hasattr(model, "effect"):
        scores = model.effect(X)
    else:
        raise TypeError(f"Unsupported model type: {type(model)!r}")
    return float(np.asarray(scores).ravel()[0])


def _stored_score_threshold() -> float | None:
    """Threshold chosen on calibration. Missing means the fare is not changed."""
    metrics = _MODEL_STATE.get("metrics") or {}
    value = metrics.get("score_threshold")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recommend_treatment(uplift_score: float, threshold: float | None = None) -> str:
    """Map the surcharge effect to one action.

    The score is the change in driver-acceptance probability from adding the
    surcharge to the base fare. It is not a discount.

    ``SURCHARGE``: the surcharge raises acceptance.
    ``KEEP_QUOTE``: the score is inside the threshold, so the quote stands.
    ``NO_SURCHARGE``: the score says the surcharge lowers acceptance.
    DPE charges the base fare only when ``ranking_supports_decision`` is true.
    """
    thr = threshold if threshold is not None else float(_MODEL_STATE.get("uplift_threshold", 0.05))
    if uplift_score > thr:
        return "SURCHARGE"
    if uplift_score < -thr:
        return "NO_SURCHARGE"
    return "KEEP_QUOTE"


def apply_model_payload(payload: dict[str, Any]) -> None:
    """Install a loaded payload into module-level state."""
    _MODEL_STATE["model"] = payload["model"]
    _MODEL_STATE["feature_columns"] = payload.get("feature_columns", FEATURE_COLUMNS)
    _MODEL_STATE["model_name"] = payload.get("model_name")
    _MODEL_STATE["source"] = payload.get("source", "unknown")
    _MODEL_STATE["uplift_threshold"] = float(
        payload.get("uplift_threshold", load_config().api.uplift_threshold)
    )
    _MODEL_STATE["metrics"] = payload.get("metrics") or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained model at startup."""
    cfg = load_config()
    path = _resolve_model_path(cfg.api.model_path)
    try:
        payload = load_model_artifact(path)
        apply_model_payload(payload)
        print(f"Loaded model '{_MODEL_STATE['model_name']}' from {path}")
        print(f"Source: {_MODEL_STATE['source']}, Feature columns: {_MODEL_STATE['feature_columns']}")
    except FileNotFoundError:
        print(
            f"WARNING: model artifact not found at {path}. "
            "Run `python scripts/train.py` before serving predictions."
        )
        _MODEL_STATE["model"] = None
    yield
    _MODEL_STATE["model"] = None


app = FastAPI(
    title="Causal Pricing & Uplift Engine",
    version="0.1.0",
    description="FastAPI service providing Causal ML Uplift treatment decisions",
    lifespan=lifespan,
)


class PredictUpliftRequest(BaseModel):
    driver_id: str
    features: dict[str, float] = Field(
        ...,
        examples=[
            {
                "distance_km": 7.0,
                "duration_sec": 900.0,
                "hour_of_day": 18.0,
                "past_trips": 12.0,
                "avg_surge": 1.25,
            }
        ],
    )


class PredictUpliftResponse(BaseModel):
    driver_id: str
    uplift_score: float
    recommended_treatment: str
    ranking_supports_decision: bool
    score_threshold: float | None = None
    model_name: str | None = None


@app.get("/health")
def health_check() -> dict[str, Any]:
    status = "ok" if _MODEL_STATE.get("model") is not None else "degraded"
    return {
        "status": status if status == "ok" else "ok",  # health always ok for liveness
        "service": "causal-pricing-engine",
        "model_loaded": "true" if _MODEL_STATE.get("model") is not None else "false",
        "model_name": str(_MODEL_STATE.get("model_name") or ""),
        "source": str(_MODEL_STATE.get("source") or ""),
        "feature_columns": _MODEL_STATE.get("feature_columns", []),
        "ranking_supports_decision": read_ranking_supports_decision(_MODEL_STATE.get("metrics")),
    }


import math
import time


@app.post("/predict_uplift", response_model=PredictUpliftResponse)
def predict_uplift(request: PredictUpliftRequest) -> PredictUpliftResponse:
    if _MODEL_STATE.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train with `python scripts/train.py` first.",
        )

    for k, v in request.features.items():
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                raise ValueError()
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{k}' has invalid non-finite value: {v}",
            )

    t0 = time.perf_counter()
    uplift_score = score_uplift(request.features)
    treatment = recommend_treatment(uplift_score)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[CPE API] /predict_uplift latency: {dt_ms:.2f} ms")

    return PredictUpliftResponse(
        driver_id=request.driver_id,
        uplift_score=uplift_score,
        recommended_treatment=treatment,
        ranking_supports_decision=read_ranking_supports_decision(_MODEL_STATE.get("metrics")),
        score_threshold=_stored_score_threshold(),
        model_name=_MODEL_STATE.get("model_name"),
    )

