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
    """On synthetic data with known positive TE segment, mean uplift should be higher there on holdout."""
    from sklearn.model_selection import train_test_split

    df = generate_uplift_dataset(n=2000, random_state=7)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["accepted"].to_numpy()
    t = df["treatment"].to_numpy()
    seg = df["segment"].to_numpy()

    X_tr, X_te, y_tr, y_te, t_tr, t_te, seg_tr, seg_te = train_test_split(
        X, y, t, seg, test_size=0.3, random_state=7, stratify=t
    )

    model = TLearner(base_estimator=GradientBoostingClassifier(n_estimators=30, random_state=7))
    model.fit(X_tr, y_tr, t_tr)
    uplift = model.predict_uplift(X_te)

    high = uplift[seg_te == 2.0]
    low = uplift[seg_te == 0.0]
    # Persuadables should have higher predicted uplift than sleeping dogs on holdout
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
    y = df["accepted"].to_numpy()
    t = df["treatment"].to_numpy()

    # Holdout only — avoid train=test optimism for metric checks
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, y, t, test_size=0.35, random_state=3, stratify=t
    )
    model = TLearner(base_estimator=LogisticRegression(max_iter=500))
    model.fit(X_tr, y_tr, t_tr)
    uplift = model.predict_uplift(X_te)

    qini = qini_auc_score(y_te, uplift, t_te)
    u_k = uplift_at_k(y_te, uplift, t_te, k=0.3)
    assert np.isfinite(qini)
    assert np.isfinite(u_k)
    # Normalized coefficient should be on a human scale (not thousands)
    assert -1.5 <= qini <= 1.5

    rng = np.random.default_rng(0)
    random_u = rng.normal(size=len(y_te))
    qini_random = qini_auc_score(y_te, random_u, t_te)
    # Model should beat random on structured synthetic HTE (soft margin)
    assert qini > qini_random - 0.05


def test_artifact_records_flag_threshold_and_false_override_rate(tmp_path: Path):
    import joblib

    from src.evaluation.metrics import false_override_rate, holdout_decision
    from src.evaluation.protocol import FALSE_OVERRIDE_RATE_MAX

    scores = np.array([-0.2, -0.08, 0.1])
    truth = np.array([0.25, -0.12, 0.0])
    rate = false_override_rate(scores, truth, threshold=0.05)
    decision = holdout_decision(
        0.3,
        false_override_rate=rate,
        false_override_rate_max=FALSE_OVERRIDE_RATE_MAX,
    )
    path = tmp_path / "model.joblib"
    joblib.dump({"metrics": decision}, path)
    metrics = joblib.load(path)["metrics"]
    assert "ranking_supports_decision" in metrics
    assert metrics["false_override_rate_max"] == pytest.approx(0.10)
    assert metrics["false_override_rate"] == pytest.approx(0.5)
    assert metrics["ranking_supports_decision"] is False


def test_holdout_splits_into_disjoint_calibration_and_test():
    from src.evaluation.metrics import split_train_calibration_test

    treatment = np.array([0, 1] * 200)
    train_idx, cal_idx, test_idx = split_train_calibration_test(
        len(treatment), treatment, random_state=42
    )
    assert len(set(train_idx) & set(cal_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(cal_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(cal_idx) + len(test_idx) == len(treatment)
    assert abs(len(cal_idx) - len(test_idx)) <= 1
    assert set(treatment[cal_idx]) == {0, 1}
    assert set(treatment[test_idx]) == {0, 1}


def test_ranking_flag_is_fixed_at_training_and_only_read_at_serve():
    from src.evaluation.metrics import holdout_decision, read_ranking_supports_decision

    below = holdout_decision(-0.003)
    above = holdout_decision(0.2)
    assert below["ranking_supports_decision"] is False
    assert above["ranking_supports_decision"] is True
    assert above["false_override_check"] == "not_applied"

    # A measured rate does not change the flag until a maximum is supplied.
    recorded = holdout_decision(0.2, false_override_rate=0.9)
    assert recorded["ranking_supports_decision"] is True
    assert recorded["false_override_check"] == "not_applied"
    assert recorded["false_override_rate"] == pytest.approx(0.9)

    blocked = holdout_decision(0.2, false_override_rate=0.4, false_override_rate_max=0.1)
    allowed = holdout_decision(0.2, false_override_rate=0.05, false_override_rate_max=0.1)
    at_cap = holdout_decision(0.2, false_override_rate=0.10, false_override_rate_max=0.10)
    missing = holdout_decision(0.2, false_override_rate=None, false_override_rate_max=0.10)
    assert blocked["ranking_supports_decision"] is False
    assert blocked["false_override_check"] == "fail"
    assert allowed["ranking_supports_decision"] is True
    assert allowed["false_override_check"] == "pass"
    assert at_cap["ranking_supports_decision"] is True
    assert missing["ranking_supports_decision"] is False
    assert missing["false_override_check"] == "undefined"

    # Serve does not turn qini_low back into a flag.
    assert read_ranking_supports_decision({"qini_low": 0.3}) is False
    assert read_ranking_supports_decision({"qini_low": -1.0, "ranking_supports_decision": True}) is True
    assert read_ranking_supports_decision(None) is False


def test_qini_beats_random_when_the_planted_effect_is_large():
    """A large planted effect must beat random scores, with the interval off zero.

    The portfolio generator plants +0.25 and -0.12. On that draw the Qini
    interval covers 0, so a test that the portfolio model beats one random
    ranking will flicker. This draw plants ±0.45. The margin is wide on purpose.
    """
    from sklearn.model_selection import train_test_split

    from src.evaluation.metrics import qini_bootstrap_interval, qini_random_interval

    rng = np.random.default_rng(11)
    n = 4000
    past_trips = rng.normal(size=n)
    distance_km = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    planted = np.where(past_trips > 0.0, 0.45, -0.45)
    accepted = rng.binomial(1, np.clip(0.40 + treatment * planted, 0.02, 0.98))
    features = np.column_stack([past_trips, distance_km])

    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        features, accepted, treatment, test_size=0.3, random_state=11, stratify=treatment
    )
    model = TLearner(
        base_estimator=GradientBoostingClassifier(
            n_estimators=40, max_depth=2, random_state=11
        )
    )
    model.fit(X_tr, y_tr, t_tr)
    scored = model.predict_uplift(X_te)
    interval = qini_bootstrap_interval(y_te, scored, t_te, n_boot=50, seed=11)
    null = qini_random_interval(y_te, t_te, n_draws=50, seed=13)

    assert interval["low"] > 0.20
    assert interval["low"] > null["high"] + 0.15


def test_qini_normalized_oracle_and_null():
    """Normalized Qini: random ≈ 0, oracle ≈ 1, unnormalized not O(n²)."""
    df = generate_uplift_dataset(n=1200, random_state=7)
    y = df["accepted"].to_numpy()
    t = df["treatment"].to_numpy()
    # Oracle score used in denominator definition
    oracle = y * (2.0 * t - 1.0)
    q_oracle = qini_auc_score(y, oracle, t, normalize=True)
    assert q_oracle == pytest.approx(1.0, abs=1e-9)

    rng = np.random.default_rng(1)
    q_rand = qini_auc_score(y, rng.normal(size=len(y)), t, normalize=True)
    assert abs(q_rand) < 0.25  # small-sample noise around 0

    # Unnormalized area over fraction is O(n * rate), not O(n²) like old metric
    q_raw = qini_auc_score(y, oracle, t, normalize=False)
    assert abs(q_raw) < len(y)  # previously was ~n² scale (~1e6)
    assert abs(q_raw) > 1.0  # still a real positive area for structured data


def test_qini_interval_is_a_distribution_not_one_draw():
    from src.evaluation.metrics import qini_bootstrap_interval, qini_random_interval

    df = generate_uplift_dataset(n=400, random_state=3)
    y = df["accepted"].to_numpy()
    t = df["treatment"].to_numpy()
    scores = df["true_uplift"].to_numpy()
    interval = qini_bootstrap_interval(y, scores, t, n_boot=30, seed=0)
    null = qini_random_interval(y, t, n_draws=30, seed=1)
    assert interval["n_boot"] >= 20
    assert interval["low"] <= interval["high"]
    assert null["n_draws"] == 30
    assert null["low"] <= null["mean"] <= null["high"]


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
        "driver_id",
        "treatment",
        "accepted",
        "revenue",
        *FEATURE_COLUMNS,
    }
    assert required.issubset(df.columns)
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["accepted"].unique()).issubset({0, 1})
    assert df["revenue"].dtype.kind == "f"
    assert "price" not in FEATURE_COLUMNS
    assert "surge_bonus" not in df.columns
    fare = df["base_fare"] + df["treatment"] * df["surcharge_rub"]
    expected_revenue = fare.where(df["accepted"] == 1, 0.0)
    assert np.allclose(df["revenue"], expected_revenue)

    # Empirical ATE by segment
    def ate(segment_val: float) -> float:
        sub = df[df["segment"] == segment_val]
        y1 = sub.loc[sub["treatment"] == 1, "accepted"].mean()
        y0 = sub.loc[sub["treatment"] == 0, "accepted"].mean()
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
    """YAML load path exposes model / experiment / api sections."""
    from pydantic_settings import BaseSettings

    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg, BaseSettings)
    assert cfg.model.base_learner
    assert cfg.model.n_estimators > 0
    assert isinstance(cfg.model.learning_rate, float)
    assert 0 < cfg.experiment.alpha < 1
    assert 0 < cfg.experiment.power < 1
    assert cfg.experiment.mde > 0
    assert cfg.api.host
    assert cfg.api.port > 0


def test_load_config_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch):
    """Shipped load_config must apply CPE_* env over YAML defaults."""
    # Baseline from YAML (no CPE_ env)
    for key in list(__import__("os").environ):
        if key.startswith("CPE_"):
            monkeypatch.delenv(key, raising=False)

    baseline = load_config()
    assert baseline.api.port == 8100
    assert baseline.api.host == "0.0.0.0"

    assert baseline.model.base_learner == "logistic_regression"

    monkeypatch.setenv("CPE_API__PORT", "9123")
    monkeypatch.setenv("CPE_API__HOST", "127.0.0.1")
    monkeypatch.setenv("CPE_MODEL__BASE_LEARNER", "gradient_boosting")
    monkeypatch.setenv("CPE_MODEL__N_ESTIMATORS", "42")
    monkeypatch.setenv("CPE_MODEL__LEARNING_RATE", "0.05")
    monkeypatch.setenv("CPE_EXPERIMENT__ALPHA", "0.01")
    monkeypatch.setenv("CPE_EXPERIMENT__POWER", "0.9")
    monkeypatch.setenv("CPE_EXPERIMENT__MDE", "0.03")

    overridden = load_config()
    assert overridden.api.port == 9123
    assert overridden.api.host == "127.0.0.1"
    assert overridden.model.base_learner == "gradient_boosting"
    assert overridden.model.n_estimators == 42
    assert overridden.model.learning_rate == pytest.approx(0.05)
    assert overridden.experiment.alpha == pytest.approx(0.01)
    assert overridden.experiment.power == pytest.approx(0.9)
    assert overridden.experiment.mde == pytest.approx(0.03)
    # Unrelated YAML field still present
    assert overridden.api.model_path

    # Cleanup is via monkeypatch teardown; also assert clear restores YAML
    monkeypatch.delenv("CPE_API__PORT", raising=False)
    monkeypatch.delenv("CPE_API__HOST", raising=False)
    monkeypatch.delenv("CPE_MODEL__BASE_LEARNER", raising=False)
    monkeypatch.delenv("CPE_MODEL__N_ESTIMATORS", raising=False)
    monkeypatch.delenv("CPE_MODEL__LEARNING_RATE", raising=False)
    monkeypatch.delenv("CPE_EXPERIMENT__ALPHA", raising=False)
    monkeypatch.delenv("CPE_EXPERIMENT__POWER", raising=False)
    monkeypatch.delenv("CPE_EXPERIMENT__MDE", raising=False)

    restored = load_config()
    assert restored.api.port == baseline.api.port
    assert restored.model.base_learner == baseline.model.base_learner


def test_config_module_uses_pydantic_settings():
    """Static/source contract: root settings subclass BaseSettings."""
    import inspect

    import src.config as config_mod
    from pydantic_settings import BaseSettings

    assert issubclass(config_mod.AppConfig, BaseSettings)
    source = inspect.getsource(config_mod)
    assert "pydantic_settings" in source or "from pydantic_settings" in source
    assert "BaseSettings" in source


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
    y = df["accepted"].to_numpy()
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

    treatment = recommend_treatment(score, threshold=0.05)
    assert treatment in {"SURCHARGE", "KEEP_QUOTE", "NO_SURCHARGE"}

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


def test_train_selects_best_among_all_candidates_including_dml(monkeypatch, tmp_path):
    """Shipped train_and_select must pick max Qini across T/S/X + DML (not exclude DML)."""
    import scripts.train as train_mod
    from src.config import AppConfig, ApiConfig, DataConfig, ExperimentConfig, ModelConfig

    # Faster run: small n; write artifact into tmp so we don't clobber only via real path
    small_cfg = AppConfig(
        model=ModelConfig(base_learner="logistic_regression", n_estimators=20, random_state=0),
        experiment=ExperimentConfig(),
        api=ApiConfig(model_path=str(tmp_path / "model.joblib"), uplift_threshold=0.05),
        data=DataConfig(n_samples=600, random_state=0),
    )
    monkeypatch.setattr(train_mod, "load_config", lambda path=None: small_cfg)
    monkeypatch.setattr(train_mod, "_ROOT", tmp_path)

    result = train_mod.train_and_select(random_state=0, include_all=True)

    metrics = result["metrics"]
    assert set(metrics.keys()) >= {"t_learner", "s_learner", "x_learner", "dml"}

    expected_best = max(metrics, key=lambda k: metrics[k]["qini_auc"])
    assert result["best_name"] == expected_best, (
        f"best_name={result['best_name']} but max Qini is {expected_best} "
        f"({ {k: v['qini_auc'] for k, v in metrics.items()} })"
    )

    artifact = Path(result["model_path"])
    assert artifact.exists() and artifact.stat().st_size > 0
    payload = joblib.load(artifact)
    assert payload["model_name"] == expected_best
    assert payload["model"] is not None

    # API scoring path must work for whichever won (predict_uplift or effect)
    from src.api.main import apply_model_payload, score_uplift, _MODEL_STATE

    apply_model_payload(payload)
    features = {c: 1.0 for c in FEATURE_COLUMNS}
    score = score_uplift(features)
    assert np.isfinite(score)
    _MODEL_STATE["model"] = None


def test_predict_uplift_nan_validation(tmp_path: Path):
    """Test that API endpoint rejects NaN/Inf/null feature values with 422 status code."""
    from fastapi.testclient import TestClient
    from src.api.main import app, apply_model_payload, _MODEL_STATE

    model_payload = {
        "model_name": "TestLearner",
        "model": TLearner(base_estimator=GradientBoostingRegressor()).fit(
            X=np.ones((10, len(FEATURE_COLUMNS))),
            treatment=np.array([0, 1] * 5),
            y=np.array([10.0, 20.0] * 5),
        ),
        "feature_columns": FEATURE_COLUMNS,
    }
    apply_model_payload(model_payload)

    client = TestClient(app)
    resp = client.post(
        "/predict_uplift",
        content='{"driver_id": "d1", "features": {"past_trips": null}}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    _MODEL_STATE["model"] = None




