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
from src.data.dpe_connector import describe_switchback, load_dpe_data, resolve_dpe_db_path
from src.data.synthetic import (
    FEATURE_COLUMNS,
    format_calibration,
    generate_uplift_dataset,
    summarize_calibration,
)
from src.evaluation.metrics import qini_bootstrap_interval, qini_random_interval, uplift_at_k

DEFAULT_THRESHOLD = 0.05
FEATURE_COLS = list(FEATURE_COLUMNS)


@dataclass(frozen=True)
class ScenarioPick:
    """One demo scenario backed by a real scored row."""

    label: str
    role: str  # persuadable | neutral | sleeping_dog
    features: dict[str, float]
    uplift: float
    row_index: int
    true_uplift: float | None = None


def load_demo_frame(source: str = "auto") -> tuple[pd.DataFrame, str]:
    """Load demo training data.

    ``synthetic`` and ``auto`` randomize an additive surcharge per offer.
    ``dpe`` prints the switchback design and does not return a training frame.
    """
    source = source.lower().strip()
    if source == "dpe":
        db_path = resolve_dpe_db_path()
        if not db_path.exists():
            raise FileNotFoundError(f"DPE DB not found at {db_path}")
        frame = load_dpe_data(db_path=db_path)
        raise SwitchbackLog(describe_switchback(frame)["report"])

    df = generate_uplift_dataset(n=3000, random_state=42)
    note = (
        "synthetic n=3000. Treatment is an additive surcharge versus the base fare, "
        "randomized per offer. Outcome is driver acceptance."
    )
    return df, note


class SwitchbackLog(Exception):
    """Raised when a caller asks the demo to train on the DPE hour-arm log."""


def pick_honest_scenarios(
    X: np.ndarray,
    uplift: np.ndarray,
    feature_cols: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    true_uplift: np.ndarray | None = None,
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

    def _planted(i: int) -> float | None:
        if true_uplift is None:
            return None
        return float(np.asarray(true_uplift, dtype=float).ravel()[i])

    return [
        ScenarioPick(
            label="Highest score — surcharge looks helpful on this row",
            role="persuadable",
            features=_row(idx_pos),
            uplift=u_pos,
            row_index=idx_pos,
            true_uplift=_planted(idx_pos),
        ),
        ScenarioPick(
            label="Score nearest zero",
            role="neutral",
            features=_row(idx_neu),
            uplift=float(uplift[idx_neu]),
            row_index=idx_neu,
            true_uplift=_planted(idx_neu),
        ),
        ScenarioPick(
            label="Lowest score — surcharge looks harmful on this row",
            role="sleeping_dog",
            features=_row(idx_neg),
            uplift=u_neg,
            row_index=idx_neg,
            true_uplift=_planted(idx_neg),
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


def _action_line(treatment: str) -> str:
    if treatment == "SURCHARGE":
        return "Act: keep the additive surcharge"
    if treatment == "NO_SURCHARGE":
        return "Act: charge the base fare (no surcharge)"
    return "Act: keep the quote (score inside the threshold)"


def run_demo(source: str = "synthetic", threshold: float = DEFAULT_THRESHOLD) -> int:
    print("=" * 64)
    print("  Causal Pricing Engine — surcharge versus base fare")
    print("=" * 64)
    print(
        "The three rows below are the highest score, a score near zero, and the\n"
        "lowest score on the holdout. They are not a discovered segment.\n"
    )

    try:
        df, source_label = load_demo_frame(source=source)
    except SwitchbackLog as exc:
        print(str(exc))
        print("[*] The demo model is not fit on this log.")
        return 0
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(f"[*] Data: {source_label}")

    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.3, random_state=42, stratify=df["treatment"].to_numpy()
    )
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["accepted"].to_numpy(dtype=int)
    t = df["treatment"].to_numpy(dtype=int)
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    t_train, t_val = t[idx_train], t[idx_val]

    print("[*] Training T-Learner (GradientBoostingClassifier)...")
    model = TLearner(
        base_estimator=GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
    )
    model.fit(X_train, y_train, t_train)

    val_uplift = model.predict_uplift(X_val)
    qini = qini_bootstrap_interval(y_val, val_uplift, t_val, n_boot=200, seed=0)
    qini_null = qini_random_interval(y_val, t_val, n_draws=200, seed=1)
    u_at_30 = uplift_at_k(y_val, val_uplift, t_val, k=0.3)
    print(
        f"[*] Holdout Qini: {qini['point']:+.4f}  "
        f"95% bootstrap [{qini['low']:+.4f}, {qini['high']:+.4f}] "
        f"({qini['n_boot']} resamples of this one split)"
    )
    if qini["low"] < 0 < qini["high"]:
        print("[*] That interval contains 0. The point estimate on this split is not separated from noise.")
    print(
        f"[*] Random-score Qini: mean {qini_null['mean']:+.4f}  "
        f"95% [{qini_null['low']:+.4f}, {qini_null['high']:+.4f}] "
        f"over {qini_null['n_draws']} draws. One draw is not a null."
    )
    print(f"[*] Holdout Uplift@30%: {u_at_30:+.4f}")
    print(f"[*] Decision threshold: ±{threshold}")
    calibration = summarize_calibration(
        df["segment"].to_numpy()[idx_val],
        df["true_uplift"].to_numpy()[idx_val],
        val_uplift,
    )
    print("[*] Calibration of the mean score against the planted effect:")
    print(format_calibration(calibration))

    try:
        scenarios = pick_honest_scenarios(
            X_val,
            val_uplift,
            FEATURE_COLS,
            threshold=threshold,
            true_uplift=df["true_uplift"].to_numpy()[idx_val],
        )
        assert_scenario_honesty(scenarios, threshold=threshold)
    except (RuntimeError, AssertionError) as exc:
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
            "metrics": {
                "qini_auc": float(qini["point"]),
                "qini_low": float(qini["low"]),
                "qini_high": float(qini["high"]),
                "uplift_at_k": float(u_at_30),
            },
            "uplift_threshold": float(threshold),
        },
        artifact_path,
    )
    print(f"[*] Saved artifact: {artifact_path}")

    print(
        "\n### Extreme holdout scores\n"
        "The planted value on the row is the segment constant, not a target the\n"
        "row was chosen to match. A large gap is miscalibration.\n"
    )
    print(
        "| Role | τ̂ | planted on this row | past_trips | distance_km | Action | Policy |"
    )
    print("|:---|:---:|:---:|:---:|:---:|:---|:---|")

    for sc in scenarios:
        treatment = recommend_treatment(sc.uplift, threshold=threshold)
        if sc.role == "persuadable" and treatment != "SURCHARGE":
            print(f"[FAIL] Highest score got treatment={treatment}")
            return 1
        if sc.role == "sleeping_dog" and treatment != "NO_SURCHARGE":
            print(f"[FAIL] Lowest score got treatment={treatment}")
            return 1

        feats = sc.features
        planted = "n/a" if sc.true_uplift is None else f"{sc.true_uplift:+.2f}"
        print(
            f"| **{sc.role}** | {sc.uplift:+.4f} | {planted} | "
            f"{feats.get('past_trips', 0):.0f} | {feats.get('distance_km', 0):.1f} | "
            f"`{treatment}` | {_action_line(treatment)} |"
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
        default="synthetic",
        help="synthetic (default) randomizes the surcharge. dpe prints the switchback design and does not train.",
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
