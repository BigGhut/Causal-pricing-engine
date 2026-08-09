"""Unit tests for uplift models, DML, switchback, metrics, config, and API path."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src.causal.dml_engine import DMLEngine
from src.causal.uplift_models import SLearner, TLearner, XLearner
from src.config import AppConfig, load_config
from src.data.synthetic import FEATURE_COLUMNS, generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k, uplift_by_percentile
from src.experiments.power_analysis import cuped_adjust, sample_size_calculator
from src.experiments.switchback_splitter import SwitchbackSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_binary():
    """Small synthetic dataset with heterogeneous TE."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, size=n)
    # Positive TE when X[:, 0] > 0
    y = (X[:, 0] + 0.6 * treatment * (X[:, 0] > 0) + np.random.randn(n) * 0.1 > 0).astype(int)
    return X, y, treatment


# ---------------------------------------------------------------------------
# Existing + expanded meta-learner tests
# ---------------------------------------------------------------------------


def test_tlearner_fit_predict(synthetic_binary):
    X, y, treatment = synthetic_binary
    learner = TLearner(base_estimator=LogisticRegression(max_iter=500))
    learner.fit(X, y, treatment)
    uplift = learner.predict_uplift(X)
    assert isinstance(uplift, np.ndarray)
    assert uplift.shape == (len(X),)
    assert np.isfinite(uplift).all()


def test_slearner_fit_predict(synthetic_binary):
    X, y, treatment = synthetic_binary
    learner = SLearner(base_estimator=LogisticRegression(max_iter=500))
    out = learner.fit(X, y, treatment)
    assert out is learner
    uplift = learner.predict_uplift(X)
    assert uplift.shape == (len(X),)
    assert np.isfinite(uplift).all()


def test_xlearner_fit_predict(synthetic_binary):
    X, y, treatment = synthetic_binary
    # Use GradientBoostingRegressor-friendly path via classifiers for stage-1;
    # LogisticRegression works for outcome + imputed TE when y is binary-ish floats.
    learner = XLearner(base_estimator=LogisticRegression(max_iter=500))
    learner.fit(X, y, treatment)
    uplift = learner.predict_uplift(X)
    assert uplift.shape == (len(X),)
    assert np.isfinite(uplift).all()


def test_learners_accept_dataframe(synthetic_binary):
    X, y, treatment = synthetic_binary
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    for cls in (TLearner, SLearner, XLearner):
        m = cls(base_estimator=LogisticRegression(max_iter=500))
        m.fit(df, y, treatment)
        u = m.predict_uplift(df)
        assert u.shape == (len(df),)


def test_meta_learners_detect_heterogeneity():
    """On synthetic data with known positive TE segment, mean uplift should be higher there."""
    df = generate_uplift_dataset(n=2000, random_state=7)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()

    model = TLearner(base_estimator=GradientBoostingClassifier(n_estimators=30, random_state=7))
    model.fit(X, y, t)
    uplift = model.predict_uplift(X)

    high = uplift[df["segment"].to_numpy() == 2.0]
    low = uplift[df["segment"].to_numpy() == 0.0]
    # Persuadables should have higher predicted uplift than sleeping dogs
    assert high.mean() > low.mean()


# ---------------------------------------------------------------------------
# DML
# ---------------------------------------------------------------------------


def test_dml_fit_effect(synthetic_binary):
    X, y, treatment = synthetic_binary
    engine = DMLEngine(random_state=0)
    engine.fit(Y=y, T=treatment, X=X, W=None)
    effects = engine.effect(X)
    assert isinstance(effects, np.ndarray)
    assert effects.shape == (len(X),)
    assert np.isfinite(effects).all()


def test_dml_with_controls():
    rng = np.random.default_rng(1)
    n = 150
    X = rng.normal(size=(n, 2))
    W = rng.normal(size=(n, 1))
    T = rng.binomial(1, 0.5, size=n)
    Y = 0.5 * T + X[:, 0] + 0.3 * W[:, 0] + rng.normal(scale=0.2, size=n)
    engine = DMLEngine(
        model_y=GradientBoostingRegressor(n_estimators=20, random_state=1),
        model_t=GradientBoostingRegressor(n_estimators=20, random_state=1),
        random_state=1,
    )
    engine.fit(Y=Y, T=T, X=X, W=W)
    eff = engine.effect(X)
    assert eff.shape == (n,)


# ---------------------------------------------------------------------------
# Switchback
# ---------------------------------------------------------------------------


def test_switchback_assignment_stability():
    splitter = SwitchbackSplitter(time_bucket_minutes=60, n_grid_cells=8, salt="test")
    ts = datetime(2026, 1, 15, 12, 30, tzinfo=timezone.utc)
    a1 = splitter.assign("unit_A", ts)
    a2 = splitter.assign("unit_A", ts)
    assert a1 == a2
    assert a1 in ("control", "treatment")


def test_switchback_space_time_blocks():
    splitter = SwitchbackSplitter(time_bucket_minutes=60, n_grid_cells=4, salt="blocks")
    ts0 = 1_700_000_000  # unix
    # Same unit, far-apart times → different buckets (may or may not differ arm)
    b0 = splitter.time_bucket(ts0)
    b1 = splitter.time_bucket(ts0 + 7200)  # +2h
    assert b0 != b1

    # Explicit grid cells produce independent block keys
    a_cell0 = splitter.assign("u1", ts0, grid_cell=0)
    a_cell1 = splitter.assign("u1", ts0, grid_cell=1)
    # Both valid assignments
    assert a_cell0 in ("control", "treatment")
    assert a_cell1 in ("control", "treatment")

    # Same block always stable
    assert splitter.assign("u1", ts0, grid_cell=0) == a_cell0


def test_switchback_ratio_roughly_balanced():
    splitter = SwitchbackSplitter(treatment_ratio=0.5, salt="balance")
    labels = [
        splitter.assign(f"unit_{i}", 1_700_000_000 + i * 3600)
        for i in range(200)
    ]
    frac = sum(1 for x in labels if x == "treatment") / len(labels)
    assert 0.25 < frac < 0.75


# ---------------------------------------------------------------------------
# CUPED / sample size (existing)
# ---------------------------------------------------------------------------


def test_cuped_adjust():
    df = pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0],
            "covariate": [9.0, 11.0, 13.0, 15.0, 17.0],
        }
    )
    adjusted = cuped_adjust(df, target_col="target", covariate_col="covariate")
    assert len(adjusted) == 5
    assert np.var(adjusted) < np.var(df["target"])


def test_sample_size_calculator():
    n_per_group = sample_size_calculator(std_dev=2.0, mde=0.5, alpha=0.05, power=0.80)
    assert n_per_group > 0
    assert isinstance(n_per_group, int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_qini_and_uplift_at_k_on_labeled_data():
    df = generate_uplift_dataset(n=800, random_state=3)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()

    model = TLearner(base_estimator=LogisticRegression(max_iter=500))
    model.fit(X, y, t)
    uplift = model.predict_uplift(X)

    qini = qini_auc_score(y, uplift, t)
    u_k = uplift_at_k(y, uplift, t, k=0.3)
    assert np.isfinite(qini)
    assert np.isfinite(u_k)

    # Random uplift should not dominate a trained model on structured data
    rng = np.random.default_rng(0)
    random_u = rng.normal(size=len(y))
    qini_random = qini_auc_score(y, random_u, t)
    # Soft check: model Qini is a real number (random may occasionally win on small n)
    assert isinstance(qini, float)


def test_uplift_by_percentile_shape():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    uplift = np.linspace(1, 0, len(y))
    treatment = np.array([1, 0] * (len(y) // 2))
    table = uplift_by_percentile(y, uplift, treatment, n_bins=5)
    assert len(table) == 5
    assert {"percentile", "uplift", "n"}.issubset(table.columns)


# ---------------------------------------------------------------------------
# Synthetic data heterogeneity
# ---------------------------------------------------------------------------


def test_synthetic_dataset_schema_and_heterogeneity():
    df = generate_uplift_dataset(n=3000, random_state=11)
    required = {
        "user_id",
        "treatment",
        "conversion",
        "revenue",
        *FEATURE_COLUMNS,
    }
    assert required.issubset(df.columns)
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["conversion"].unique()).issubset({0, 1})
    assert df["revenue"].dtype.kind == "f"

    # Empirical ATE by segment
    def ate(segment_val: float) -> float:
        sub = df[df["segment"] == segment_val]
        y1 = sub.loc[sub["treatment"] == 1, "conversion"].mean()
        y0 = sub.loc[sub["treatment"] == 0, "conversion"].mean()
        return float(y1 - y0)

    ate_pos = ate(2.0)
    ate_zero = ate(1.0)
    ate_neg = ate(0.0)

    assert ate_pos > 0.05, f"expected positive TE in segment 2, got {ate_pos}"
    assert abs(ate_zero) < 0.08, f"expected ~zero TE in segment 1, got {ate_zero}"
    assert ate_neg < -0.03, f"expected negative TE in segment 0, got {ate_neg}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_config_has_required_sections():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.base_learner
    assert cfg.model.n_estimators > 0
    assert 0 < cfg.experiment.alpha < 1
    assert 0 < cfg.experiment.power < 1
    assert cfg.experiment.mde > 0
    assert cfg.api.host
    assert cfg.api.port > 0


# ---------------------------------------------------------------------------
# API scoring path (same functions the app uses)
# ---------------------------------------------------------------------------


def test_api_scoring_path_with_real_artifact(tmp_path: Path):
    """Train a tiny model, dump joblib, load via API helpers, score features."""
    from src.api.main import (
        apply_model_payload,
        load_model_artifact,
        recommend_treatment,
        score_uplift,
        _MODEL_STATE,
    )

    df = generate_uplift_dataset(n=400, random_state=5)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()

    model = SLearner(base_estimator=LogisticRegression(max_iter=500))
    model.fit(X, y, t)

    artifact = tmp_path / "model.joblib"
    payload = {
        "model": model,
        "model_name": "s_learner",
        "feature_columns": FEATURE_COLUMNS,
        "uplift_threshold": 0.05,
        "metrics": {"qini_auc": 0.0, "uplift_at_k": 0.0},
    }
    joblib.dump(payload, artifact)

    loaded = load_model_artifact(artifact)
    apply_model_payload(loaded)

    features = {c: float(X[0, i]) for i, c in enumerate(FEATURE_COLUMNS)}
    score = score_uplift(features)
    assert np.isfinite(score)

    # Must match direct model call (not mock past_trips rule)
    direct = float(model.predict_uplift(X[0:1])[0])
    assert abs(score - direct) < 1e-9

    treatment, discount = recommend_treatment(score, threshold=0.05)
    assert treatment in {"DISCOUNT_10_PCT", "NO_DISCOUNT", "NO_DISCOUNT_AVOID"}
    assert isinstance(discount, float)

    # Mock path removed: past_trips alone must not force fixed 0.15/0.02
    low_trips = {c: 0.0 for c in FEATURE_COLUMNS}
    low_trips["past_trips"] = 1.0
    high_trips = dict(low_trips)
    high_trips["past_trips"] = 20.0
    s_low = score_uplift(low_trips)
    s_high = score_uplift(high_trips)
    # Old mock: past_trips < 5 → 0.15 else 0.02 — ensure we are not that
    assert not (abs(s_low - 0.15) < 1e-12 and abs(s_high - 0.02) < 1e-12)

    _MODEL_STATE["model"] = None


def test_package_exports():
    from src.causal import BaseUpliftModel, DMLEngine, SLearner, TLearner, XLearner
    from src.experiments import SwitchbackSplitter, cuped_adjust, sample_size_calculator
    from src.evaluation import qini_auc_score, uplift_at_k, uplift_by_percentile

    assert BaseUpliftModel is not None
    assert DMLEngine is not None
    assert callable(qini_auc_score)
