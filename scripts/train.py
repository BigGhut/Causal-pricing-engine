"""Training pipeline: fit uplift models, evaluate, serialize the best."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as ``python scripts/train.py``
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.causal.dml_engine import DMLEngine
from src.causal.uplift_models import BaseUpliftModel, SLearner, TLearner, XLearner
from src.config import load_config
from src.data.synthetic import FEATURE_COLUMNS, generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k


def _build_base_estimator(name: str, n_estimators: int, learning_rate: float, random_state: int):
    """Construct an sklearn-compatible classifier from config name."""
    name = name.lower()
    if name in {"logistic_regression", "logreg", "lr"}:
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name in {"gradient_boosting", "gbm", "gbc"}:
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                random_seed=random_state,
                verbose=0,
            )
        except ImportError:
            print("CatBoost not installed; falling back to GradientBoostingClassifier")
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
    # Default
    return LogisticRegression(max_iter=1000, random_state=random_state)


def train_and_select(random_state: int | None = None) -> dict:
    """Generate data, train candidates, pick best by Qini AUC, dump artifact.

    Returns:
        Dict with metrics and paths for programmatic checks.
    """
    cfg = load_config()
    rs = random_state if random_state is not None else cfg.data.random_state

    print(f"Generating synthetic uplift dataset (n={cfg.data.n_samples})...")
    df = generate_uplift_dataset(n=cfg.data.n_samples, random_state=rs)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["conversion"].to_numpy(dtype=int)
    treatment = df["treatment"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, treatment, test_size=0.3, random_state=rs, stratify=treatment
    )

    base = _build_base_estimator(
        cfg.model.base_learner,
        cfg.model.n_estimators,
        cfg.model.learning_rate,
        cfg.model.random_state,
    )

    candidates: dict[str, BaseUpliftModel | DMLEngine] = {
        "t_learner": TLearner(base_estimator=base),
        "s_learner": SLearner(base_estimator=base),
        "x_learner": XLearner(base_estimator=base),
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train, t_train)
        uplift = model.predict_uplift(X_test)
        qini = qini_auc_score(y_test, uplift, t_test)
        u_at_k = uplift_at_k(y_test, uplift, t_test, k=0.3)
        results[name] = {"qini_auc": float(qini), "uplift_at_k": float(u_at_k)}
        print(f"  {name}: Qini AUC={qini:.4f}, Uplift@k={u_at_k:.4f}")

    # DML on continuous/binary outcome
    print("Training dml_engine...")
    dml = DMLEngine(random_state=cfg.model.random_state)
    dml.fit(Y=y_train, T=t_train, X=X_train, W=None)
    dml_uplift = dml.effect(X_test)
    dml_qini = qini_auc_score(y_test, dml_uplift, t_test)
    dml_uk = uplift_at_k(y_test, dml_uplift, t_test, k=0.3)
    results["dml"] = {"qini_auc": float(dml_qini), "uplift_at_k": float(dml_uk)}
    print(f"  dml: Qini AUC={dml_qini:.4f}, Uplift@k={dml_uk:.4f}")
    candidates["dml"] = dml

    # Select best by Qini AUC among ALL candidates (T/S/X-Learner + DML).
    # API score_uplift already supports both predict_uplift and effect().
    scores = {k: v["qini_auc"] for k, v in results.items()}
    best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_model = candidates[best_name]

    artifacts_dir = _ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": results[best_name],
        "all_metrics": results,
        "uplift_threshold": cfg.api.uplift_threshold,
    }
    joblib.dump(payload, model_path)

    print("\n=== Training complete ===")
    print(f"Best model: {best_name}")
    print(f"Qini AUC:   {results[best_name]['qini_auc']:.4f}")
    print(f"Uplift@k:   {results[best_name]['uplift_at_k']:.4f}")
    print(f"Saved to:   {model_path}")

    return {
        "best_name": best_name,
        "model_path": str(model_path),
        "metrics": results,
    }


def main() -> None:
    train_and_select()


if __name__ == "__main__":
    main()
