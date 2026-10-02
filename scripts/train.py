"""Training pipeline: fit uplift models, evaluate, serialize the best."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as ``python scripts/train.py``
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.causal.dml_engine import DMLEngine
from src.causal.uplift_models import BaseUpliftModel, SLearner, TLearner, XLearner
from src.config import load_config
from src.data.dpe_connector import describe_switchback, load_dpe_data
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


def train_and_select(
    source: str = "synthetic",
    dpe_db_path: str | Path | None = None,
    random_state: int | None = None,
    learners: list[str] | None = None,
    include_all: bool = False,
    feature_mode: str = "pre_treatment",
) -> dict:
    """Fit the surcharge model on synthetic randomized offers.

    ``source='dpe'`` does not fit a model. Those rows are a switchback of two
    surge formulas, assigned by virtual hour. The function prints that design
    and returns without writing an artifact.

    ``feature_mode`` applies only to the DPE summary. The default is
    ``pre_treatment`` (no price, no surge_bonus).
    """
    cfg = load_config()
    rs = random_state if random_state is not None else cfg.data.random_state

    source_clean = source.lower()
    if source_clean == "auto":
        source_clean = "synthetic"

    if source_clean == "dpe":
        frame = load_dpe_data(db_path=dpe_db_path, feature_mode=feature_mode)
        design = describe_switchback(frame)
        print(design["report"])
        print("\nNo model artifact written. Train with --source synthetic.")
        return {"refused": True, "identified_ite": False, "design": design}

    print(
        f"Generating synthetic offers (n={cfg.data.n_samples}). "
        "Treatment is an additive surcharge versus the base fare, "
        "randomized per offer. Outcome is driver acceptance."
    )
    df = generate_uplift_dataset(n=cfg.data.n_samples, random_state=rs)
    feature_cols = FEATURE_COLUMNS

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["accepted"].to_numpy(dtype=int)
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

    # Determine which candidates to train
    if include_all:
        target_learners = ["t_learner", "s_learner", "x_learner", "dml"]
    elif learners:
        target_learners = [l.lower().strip() for l in learners]
    else:
        target_learners = ["t_learner"]

    all_possible: dict[str, BaseUpliftModel | DMLEngine] = {
        "t_learner": TLearner(base_estimator=base),
        "s_learner": SLearner(base_estimator=base),
        "x_learner": XLearner(base_estimator=base),
    }

    candidates: dict[str, BaseUpliftModel | DMLEngine] = {}
    for lname in target_learners:
        if lname in all_possible:
            candidates[lname] = all_possible[lname]
        elif lname == "dml":
            candidates["dml"] = DMLEngine(random_state=cfg.model.random_state)

    results: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        if name == "dml":
            model.fit(Y=y_train, T=t_train, X=X_train, W=None)
            uplift = model.effect(X_test)
        else:
            model.fit(X_train, y_train, t_train)
            uplift = model.predict_uplift(X_test)

        qini = qini_auc_score(y_test, uplift, t_test)
        u_at_k = uplift_at_k(y_test, uplift, t_test, k=0.3)
        results[name] = {"qini_auc": float(qini), "uplift_at_k": float(u_at_k)}
        print(f"  {name}: Qini coef={qini:+.4f}, Uplift@k={u_at_k:+.4f}")

    # Select best by Qini AUC among trained candidates
    scores = {k: v["qini_auc"] for k, v in results.items()}
    best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_model = candidates[best_name]

    artifacts_dir = _ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "source": source_clean,
        "metrics": results[best_name],
        "all_metrics": results,
        "uplift_threshold": cfg.api.uplift_threshold,
    }
    joblib.dump(payload, model_path)

    print("\n=== Training complete ===")
    print(f"Data source: {source_clean}")
    print(f"Best model:  {best_name}")
    print(f"Qini coef:   {results[best_name]['qini_auc']:+.4f}  (normalized ~[-1,1])")
    print(f"Uplift@k:    {results[best_name]['uplift_at_k']:.4f}")
    print(f"Saved to:    {model_path}")
    print(
        "Features are pre-treatment only: "
        + ", ".join(feature_cols)
        + ".\nprice and surge_bonus are not in the model. "
        "DPE calls /predict_uplift only for an additive hour that names a driver."
    )

    return {
        "best_name": best_name,
        "model_path": str(model_path),
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train uplift models and save artifact.")
    parser.add_argument(
        "--source",
        choices=["auto", "synthetic", "dpe"],
        default="synthetic",
        help="synthetic (default) randomizes the surcharge. dpe only prints the switchback design and does not fit a model. auto is synthetic.",
    )
    parser.add_argument(
        "--dpe-db-path",
        type=str,
        default=None,
        help="Optional custom path to dpe_database.db",
    )
    parser.add_argument(
        "--learners",
        type=str,
        default="t_learner",
        help="Comma-separated learners to evaluate: t_learner (default), s_learner, x_learner, dml",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Train and evaluate all available candidate models (t, s, x, dml)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["serve_parity", "pre_treatment"],
        default="pre_treatment",
        help="DPE summary only. pre_treatment (default) omits price and surge_bonus. serve_parity keeps them for inspection.",
    )
    args = parser.parse_args()

    learners_list = [s.strip() for s in args.learners.split(",")] if args.learners else None
    train_and_select(
        source=args.source,
        dpe_db_path=args.dpe_db_path,
        learners=learners_list,
        include_all=args.include_all,
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
