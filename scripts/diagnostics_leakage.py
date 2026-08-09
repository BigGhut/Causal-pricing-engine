"""Diagnostics CLI script: check propensity score overlap / leakage / Qini breakdown.

Exit 0 always (diagnostic tool).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.causal.uplift_models import TLearner
from src.data.dpe_connector import (
    DPE_FEATURE_COLUMNS,
    DPE_PRE_TREATMENT_FEATURES,
    load_dpe_data,
    resolve_dpe_db_path,
)
from src.data.synthetic import FEATURE_COLUMNS, generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k


def diagnose_synthetic() -> None:
    print("--- [Diagnostics: Synthetic Dataset] ---")
    df = generate_uplift_dataset(n=3000, random_state=42)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["conversion"].to_numpy()
    t = df["treatment"].to_numpy()

    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=t
    )

    # Propensity AUC (Treatment given X)
    clf_prop = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf_prop.fit(X_tr, t_tr)
    prop_pred = clf_prop.predict_proba(X_te)[:, 1]
    auc_t = roc_auc_score(t_te, prop_pred)
    print(f"Propensity AUC(T|X):        {auc_t:.4f} (expected ~0.50 for RCT)")

    # Model evaluation
    model = TLearner(base_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42))
    model.fit(X_tr, y_tr, t_tr)
    u_te = model.predict_uplift(X_te)

    q_model = qini_auc_score(y_te, u_te, t_te)
    rng = np.random.default_rng(42)
    q_null = qini_auc_score(y_te, rng.normal(size=len(y_te)), t_te)

    oracle = y_te * (2.0 * t_te - 1.0)
    q_oracle = qini_auc_score(y_te, oracle, t_te)

    print(f"Qini Model (normalized):   {q_model:+.4f}")
    print(f"Qini Random Null:          {q_null:+.4f}")
    print(f"Qini Oracle:               {q_oracle:+.4f}")
    print(f"Uplift@30%:                {uplift_at_k(y_te, u_te, t_te, k=0.3):+.4f}")


def diagnose_dpe() -> None:
    print("\n--- [Diagnostics: DPE Simulation Database] ---")
    db_path = resolve_dpe_db_path()
    if not db_path.exists():
        print(f"DPE database not found at {db_path} — skipping DPE diagnostics.")
        return

    # Serve parity mode (7 features)
    df_serve = load_dpe_data(db_path=db_path, feature_mode="serve_parity")
    X_serve = df_serve[DPE_FEATURE_COLUMNS].to_numpy()
    y_s = df_serve["conversion"].to_numpy()
    t_s = df_serve["treatment"].to_numpy()

    # Propensity AUC
    clf_p = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf_p.fit(X_serve, t_s)
    prop_s = clf_p.predict_proba(X_serve)[:, 1]
    auc_serve = roc_auc_score(t_s, prop_s)
    print(f"DPE (serve_parity) Propensity AUC(T|X): {auc_serve:.4f}")
    print(f"DPE Mean surge_bonus by arm: T=1 (ADDITIVE): {df_serve.loc[t_s==1, 'surge_bonus'].mean():.2f}, T=0 (MULT): {df_serve.loc[t_s==0, 'surge_bonus'].mean():.2f}")

    # Pre-treatment mode (5 features)
    df_pre = load_dpe_data(db_path=db_path, feature_mode="pre_treatment")
    X_pre = df_pre[DPE_PRE_TREATMENT_FEATURES].to_numpy()
    clf_p2 = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf_p2.fit(X_pre, t_s)
    prop_pre = clf_p2.predict_proba(X_pre)[:, 1]
    auc_pre = roc_auc_score(t_s, prop_pre)
    print(f"DPE (pre_treatment) Propensity AUC(T|X): {auc_pre:.4f}")

    # Holdout eval for pre_treatment model
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X_pre, y_s, t_s, test_size=0.3, random_state=42, stratify=t_s
    )
    model = TLearner(base_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42))
    model.fit(X_tr, y_tr, t_tr)
    u_te = model.predict_uplift(X_te)
    q_pre = qini_auc_score(y_te, u_te, t_te)
    print(f"DPE (pre_treatment) Holdout Qini:      {q_pre:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run propensity and leakage diagnostics.")
    args = parser.parse_args()
    diagnose_synthetic()
    diagnose_dpe()
    sys.exit(0)


if __name__ == "__main__":
    main()
