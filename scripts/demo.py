"""In-process happy path demo for Causal Pricing Engine.

Demonstrates data resolution (DPE SQLite or synthetic fallback), T-Learner training,
evaluation (Qini AUC), artifact serialization, and sample predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.api.main import recommend_treatment
from src.causal.uplift_models import TLearner
from src.data.dpe_connector import DPE_FEATURE_COLUMNS, load_dpe_data, resolve_dpe_db_path
from src.data.synthetic import generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k


def run_demo() -> int:
    print("=" * 64)
    print("      Causal Pricing Engine — Portfolio Demo (In-Process)")
    print("=" * 64)

    # 1. Resolve data source (DPE DB if exists, else synthetic)
    db_path = resolve_dpe_db_path()
    if db_path.exists():
        print(f"[*] Found DPE database: {db_path}")
        df = load_dpe_data(db_path=db_path)
        source_label = f"DPE SQLite ({db_path.name}, n={len(df)})"
    else:
        print("[*] DPE database not found. Generating synthetic dataset (n=2000)...")
        df = generate_uplift_dataset(n=2000, random_state=42)
        source_label = "Synthetic (n=2000)"

    feature_cols = DPE_FEATURE_COLUMNS
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["conversion"].to_numpy(dtype=int)
    t = df["treatment"].to_numpy(dtype=int)

    # 2. Split and Train T-Learner
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=t
    )

    print("[*] Training T-Learner model (GradientBoostingClassifier)...")
    base_est = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model = TLearner(base_estimator=base_est)
    model.fit(X_train, y_train, t_train)

    # 3. Evaluate model
    val_uplift = model.predict_uplift(X_val)
    qini = qini_auc_score(y_val, val_uplift, t_val)
    u_at_30 = uplift_at_k(y_val, val_uplift, t_val, k=0.3)

    print(f"[*] Validation Qini AUC:  {qini:.4f}")
    print(f"[*] Validation Uplift@30%: {u_at_30:.4f}")

    # 4. Save artifact
    artifacts_dir = _ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "model.joblib"

    payload = {
        "model": model,
        "model_name": "t_learner",
        "feature_columns": feature_cols,
        "source": source_label,
        "metrics": {"qini_auc": float(qini), "uplift_at_k": float(u_at_30)},
        "uplift_threshold": 0.05,
    }
    joblib.dump(payload, artifact_path)
    print(f"[*] Saved model artifact to: {artifact_path}")

    # 5. Score 3 representative scenarios
    scenarios = [
        {
            "name": "Persuadable (High Surge + Active Driver)",
            "features": {
                "distance_km": 7.5,
                "duration_sec": 900.0,
                "price": 320.0,
                "surge_bonus": 45.0,
                "hour_of_day": 18.0,
                "past_trips": 15.0,
                "avg_surge": 20.0,
            },
        },
        {
            "name": "Neutral (Standard Trip)",
            "features": {
                "distance_km": 5.0,
                "duration_sec": 600.0,
                "price": 200.0,
                "surge_bonus": 5.0,
                "hour_of_day": 12.0,
                "past_trips": 6.0,
                "avg_surge": 5.0,
            },
        },
        {
            "name": "Sleeping Dog (Surge Churn Risk)",
            "features": {
                "distance_km": 12.0,
                "duration_sec": 1400.0,
                "price": 480.0,
                "surge_bonus": 35.0,
                "hour_of_day": 23.0,
                "past_trips": 1.0,
                "avg_surge": 28.0,
            },
        },
    ]

    print("\n### Real-time Uplift Decision Sample\n")
    print(
        "| Scenario | Past Trips | Surge Bonus | Uplift Score | Treatment Code | Action |"
    )
    print(
        "|:---|:---:|:---:|:---:|:---|:---|"
    )

    for sc in scenarios:
        feats = sc["features"]
        x_row = np.asarray([[feats[c] for c in feature_cols]], dtype=float)
        score = float(model.predict_uplift(x_row)[0])
        treatment, discount = recommend_treatment(score, threshold=0.05)

        action_desc = (
            f"Offer {discount:.0f}% discount"
            if discount > 0
            else ("Avoid surge / override" if treatment == "NO_DISCOUNT_AVOID" else "Standard fare")
        )

        print(
            f"| {sc['name']} | {feats['past_trips']:.0f} | {feats['surge_bonus']:.1f} | "
            f"{score:+.4f} | `{treatment}` | {action_desc} |"
        )

    print("\n" + "=" * 64)
    print("Demo completed successfully.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
