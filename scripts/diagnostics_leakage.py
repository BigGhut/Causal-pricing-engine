"""Diagnostics: propensity of the randomized surcharge, and the DPE switchback design.

Does not fit an individual model on the DPE log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.causal.uplift_models import TLearner
from src.data.dpe_connector import describe_switchback, load_dpe_data, resolve_dpe_db_path
from src.data.synthetic import FEATURE_COLUMNS, format_calibration, generate_uplift_dataset, summarize_calibration
from src.evaluation.metrics import qini_bootstrap_interval, qini_random_interval, uplift_at_k


def diagnose_synthetic() -> None:
    print("--- Synthetic surcharge experiment ---")
    df = generate_uplift_dataset(n=3000, random_state=42)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["accepted"].to_numpy()
    t = df["treatment"].to_numpy()

    X_tr, X_te, y_tr, y_te, t_tr, t_te, idx_tr, idx_te = train_test_split(
        X,
        y,
        t,
        df.index.to_numpy(),
        test_size=0.3,
        random_state=42,
        stratify=t,
    )

    clf_prop = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf_prop.fit(X_tr, t_tr)
    prop_pred = clf_prop.predict_proba(X_te)[:, 1]
    auc_t = roc_auc_score(t_te, prop_pred)
    print(f"Propensity AUC(T|X): {auc_t:.4f} (near 0.50 when the surcharge is randomized)")

    model = TLearner(base_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42))
    model.fit(X_tr, y_tr, t_tr)
    u_te = model.predict_uplift(X_te)
    qini = qini_bootstrap_interval(y_te, u_te, t_te, n_boot=100, seed=0)
    null = qini_random_interval(y_te, t_te, n_draws=100, seed=1)
    print(
        f"Qini {qini['point']:+.4f} 95% [{qini['low']:+.4f}, {qini['high']:+.4f}]"
    )
    print(
        f"Random-score Qini mean {null['mean']:+.4f} "
        f"95% [{null['low']:+.4f}, {null['high']:+.4f}] over {null['n_draws']} draws"
    )
    print(f"Uplift@30%: {uplift_at_k(y_te, u_te, t_te, k=0.3):+.4f}")
    held = df.iloc[idx_te]
    print(format_calibration(summarize_calibration(
        held["segment"].to_numpy(),
        held["true_uplift"].to_numpy(),
        u_te,
    )))
    del idx_tr


def diagnose_dpe() -> None:
    print("\n--- DPE switchback log ---")
    db_path = resolve_dpe_db_path()
    if not db_path.exists():
        print(f"DPE database not found at {db_path}.")
        return

    frame = load_dpe_data(db_path=db_path, feature_mode="serve_parity")
    print(describe_switchback(frame)["report"])
    if "surge_bonus" in frame.columns:
        print("\nMean surge_bonus by arm (this column is a result of the arm):")
        for arm, sub in frame.groupby("pricing_arm"):
            print(f"  {arm}: {sub['surge_bonus'].mean():.2f}")
    print("No T-learner is fit on these rows.")


def main() -> None:
    argparse.ArgumentParser(description="Propensity and switchback diagnostics.").parse_args()
    diagnose_synthetic()
    diagnose_dpe()
    sys.exit(0)


if __name__ == "__main__":
    main()
