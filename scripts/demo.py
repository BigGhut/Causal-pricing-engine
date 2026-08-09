"""Honest in-process portfolio demo for Causal Pricing Engine.

Trains a T-Learner, then selects three *real* validation rows by predicted
uplift so scenario labels match scores:

- Persuadable  → high positive τ  (above threshold)
- Neutral      → τ closest to zero (within threshold band when possible)
- Sleeping Dog → most negative τ (below −threshold)

Hand-crafted feature vectors are intentionally not used: they previously
mislabeled rows (e.g. "Sleeping Dog" with positive uplift).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.api.main import recommend_treatment
from src.causal.uplift_models import TLearner
from src.data.dpe_connector import DPE_FEATURE_COLUMNS, load_dpe_data, resolve_dpe_db_path
from src.data.synthetic import generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k

DEFAULT_THRESHOLD = 0.05
FEATURE_COLS = list(DPE_FEATURE_COLUMNS)


@dataclass(frozen=True)
class ScenarioPick:
    """One demo scenario backed by a real scored row."""

    label: str
    role: str  # persuadable | neutral | sleeping_dog
    features: dict[str, float]
    uplift: float
    row_index: int


def load_demo_frame(source: str = "auto") -> tuple[pd.DataFrame, str]:
    """Load demo training data.

    ``auto`` prefers synthetic for guaranteed heterogeneous TE (honest
    quadrant demo). Use ``dpe`` to force simulation DB; falls back to
    synthetic with an explicit notice if DB is missing or too small.
    """
    source = source.lower().strip()
    db_path = resolve_dpe_db_path()

    if source == "synthetic":
        df = generate_uplift_dataset(n=3000, random_state=42)
        return df, "synthetic (n=3000, seeded HTE)"

    if source == "dpe":
        if not db_path.exists():
            print(f"[!] DPE DB not found at {db_path}; falling back to synthetic.")
            df = generate_uplift_dataset(n=3000, random_state=42)
            return df, "synthetic fallback (DPE DB missing)"
        df = load_dpe_data(db_path=db_path)
        return df, f"DPE SQLite ({db_path.name}, n={len(df)})"

    # auto: synthetic first for portfolio honesty / reproducibility
    # (DPE still available via --source dpe)
    df = generate_uplift_dataset(n=3000, random_state=42)
    note = "synthetic (n=3000, seeded HTE)"
    if db_path.exists():
        note += f" | DPE DB available at {db_path.name} (use --source dpe to train on it)"
    return df, note


def pick_honest_scenarios(
    X: np.ndarray,
    uplift: np.ndarray,
    feature_cols: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ScenarioPick]:
    """Pick three validation rows whose scores match scenario semantics.

    Raises:
        RuntimeError: if the model does not produce both a clear positive
            and a clear negative uplift on the scored pool (cannot demo
            honestly).
    """
    uplift = np.asarray(uplift, dtype=float).ravel()
    if len(uplift) < 3:
        raise RuntimeError("Need at least 3 scored rows to pick scenarios.")

    thr = float(threshold)
    idx_pos = int(np.argmax(uplift))
    idx_neg = int(np.argmin(uplift))
    u_pos = float(uplift[idx_pos])
    u_neg = float(uplift[idx_neg])

    if u_pos <= thr:
        raise RuntimeError(
            f"No persuadable row: max uplift={u_pos:+.4f} is not > threshold {thr}. "
            "Model/data lack positive HTE for an honest demo."
        )
    if u_neg >= -thr:
        raise RuntimeError(
            f"No sleeping-dog row: min uplift={u_neg:+.4f} is not < -threshold {-thr}. "
            "Model/data lack negative HTE for an honest demo. "
            "Try: python scripts/demo.py --source synthetic"
        )

    # Neutral: closest to zero among rows that are not the pos/neg exemplars
    # Prefer |τ| <= thr when available.
    candidates = np.ones(len(uplift), dtype=bool)
    candidates[idx_pos] = False
    candidates[idx_neg] = False
    band = candidates & (np.abs(uplift) <= thr)
    if band.any():
        pool = np.where(band)[0]
        idx_neu = int(pool[np.argmin(np.abs(uplift[pool]))])
    else:
        pool = np.where(candidates)[0]
        idx_neu = int(pool[np.argmin(np.abs(uplift[pool]))])

    def _row(i: int) -> dict[str, float]:
        return {c: float(X[i, j]) for j, c in enumerate(feature_cols)}

    return [
        ScenarioPick(
            label="Persuadable — treatment likely helps",
            role="persuadable",
            features=_row(idx_pos),
            uplift=u_pos,
            row_index=idx_pos,
        ),
        ScenarioPick(
            label="Neutral — little incremental effect",
            role="neutral",
            features=_row(idx_neu),
            uplift=float(uplift[idx_neu]),
            row_index=idx_neu,
        ),
        ScenarioPick(
            label="Sleeping Dog — treatment likely hurts",
            role="sleeping_dog",
            features=_row(idx_neg),
            uplift=u_neg,
            row_index=idx_neg,
        ),
    ]


def assert_scenario_honesty(
    scenarios: list[ScenarioPick],
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Hard checks so demo labels cannot drift from scores again."""
    by_role = {s.role: s for s in scenarios}
    thr = float(threshold)
    if by_role["persuadable"].uplift <= thr:
        raise AssertionError("Persuadable uplift must be > threshold")
    if by_role["sleeping_dog"].uplift >= -thr:
        raise AssertionError("Sleeping Dog uplift must be < -threshold")
    # Neutral should be the mildest of the three in absolute terms vs extremes
    if abs(by_role["neutral"].uplift) > abs(by_role["persuadable"].uplift):
        raise AssertionError("Neutral |uplift| should not exceed persuadable")
    if abs(by_role["neutral"].uplift) > abs(by_role["sleeping_dog"].uplift):
        raise AssertionError("Neutral |uplift| should not exceed sleeping dog")


def _action_line(treatment: str, discount: float) -> str:
    if treatment == "DISCOUNT_10_PCT" or discount > 0:
        return f"Act: offer treatment / discount ({discount:.0f}%)"
    if treatment == "NO_DISCOUNT_AVOID":
        return "Act: suppress surge / avoid treatment (Sleeping Dog)"
    return "Act: keep baseline fare (no incremental treatment)"


def run_demo(source: str = "auto", threshold: float = DEFAULT_THRESHOLD) -> int:
    print("=" * 64)
    print("  Causal Pricing Engine — Honest Portfolio Demo")
    print("=" * 64)
    print(
        "Scenarios are real validation rows ranked by predicted τ(x),\n"
        "not hand-picked feature vectors with decorative labels.\n"
    )

    df, source_label = load_demo_frame(source=source)
    print(f"[*] Data: {source_label}")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["conversion"].to_numpy(dtype=int)
    t = df["treatment"].to_numpy(dtype=int)

    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=t
    )

    print("[*] Training T-Learner (GradientBoostingClassifier)...")
    model = TLearner(
        base_estimator=GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
    )
    model.fit(X_train, y_train, t_train)

    val_uplift = model.predict_uplift(X_val)
    qini = qini_auc_score(y_val, val_uplift, t_val)  # normalized coefficient ~[-1, 1]
    u_at_30 = uplift_at_k(y_val, val_uplift, t_val, k=0.3)
    # Null bar: random scores should land near 0
    rng = np.random.default_rng(0)
    qini_null = qini_auc_score(y_val, rng.normal(size=len(y_val)), t_val)
    print(f"[*] Holdout Qini coefficient: {qini:+.4f}  (normalized; random null {qini_null:+.4f})")
    print(f"[*] Holdout Uplift@30%:       {u_at_30:+.4f}")
    print(f"[*] Decision threshold:       ±{threshold}")

    try:
        scenarios = pick_honest_scenarios(
            X_val, val_uplift, FEATURE_COLS, threshold=threshold
        )
        assert_scenario_honesty(scenarios, threshold=threshold)
    except (RuntimeError, AssertionError) as exc:
        if source != "synthetic":
            print(f"[!] Honest pick failed on this data: {exc}")
            print("[*] Retrying with synthetic HTE data...")
            return run_demo(source="synthetic", threshold=threshold)
        print(f"[FAIL] {exc}")
        return 1

    # Persist artifact (same schema as train/API)
    artifacts_dir = _ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": "t_learner",
            "feature_columns": FEATURE_COLS,
            "source": source_label,
            "metrics": {"qini_auc": float(qini), "uplift_at_k": float(u_at_30)},
            "uplift_threshold": float(threshold),
        },
        artifact_path,
    )
    print(f"[*] Saved artifact: {artifact_path}")

    print("\n### Honest uplift decisions (from holdout rows)\n")
    print(
        "| Role | τ̂ (uplift) | past_trips | surge_bonus | Treatment | Policy |"
    )
    print("|:---|:---:|:---:|:---:|:---|:---|")

    for sc in scenarios:
        treatment, discount = recommend_treatment(sc.uplift, threshold=threshold)
        # Sanity: treatment code must agree with score sign bands
        if sc.role == "persuadable" and treatment != "DISCOUNT_10_PCT":
            print(f"[FAIL] Persuadable got treatment={treatment}")
            return 1
        if sc.role == "sleeping_dog" and treatment != "NO_DISCOUNT_AVOID":
            print(f"[FAIL] Sleeping Dog got treatment={treatment}")
            return 1

        feats = sc.features
        print(
            f"| **{sc.role}** | {sc.uplift:+.4f} | "
            f"{feats.get('past_trips', 0):.0f} | {feats.get('surge_bonus', 0):.1f} | "
            f"`{treatment}` | {_action_line(treatment, discount)} |"
        )

    print("\nFeature vectors (for reproducibility):\n")
    for sc in scenarios:
        feat_str = ", ".join(f"{k}={v:.2f}" for k, v in sc.features.items())
        print(f"  [{sc.role}] {feat_str}")

    print("\n" + "=" * 64)
    print("Demo OK — labels match score signs at threshold ±"
          f"{threshold}.")
    print("=" * 64)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest CPE portfolio demo")
    parser.add_argument(
        "--source",
        choices=["auto", "synthetic", "dpe"],
        default="auto",
        help="auto=synthetic HTE (default, honest); dpe=simulation DB; synthetic=force",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Uplift decision threshold (default 0.05)",
    )
    args = parser.parse_args()
    raise SystemExit(run_demo(source=args.source, threshold=args.threshold))


if __name__ == "__main__":
    main()
