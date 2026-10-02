"""Portfolio proof: honest scenarios → live CPE HTTP → DPE policy simulation.

Produces ``docs/evidence/latest_proof.md`` with *captured* JSON (not invented).

Steps:
1. Train T-Learner on synthetic HTE (same as honest demo).
2. Pick persuadable / neutral / sleeping_dog rows by predicted τ̂.
3. Start CPE (:8100) if needed, POST ``/predict_uplift`` for each row.
4. Apply the same Sleeping Dog rule DPE uses (τ̂ < −threshold → override).
5. Optionally hit DPE ``/api/v1/search`` when ``--with-dpe`` (best-effort;
   graph features may not match the demo row, so override is not required).

Exit 0 only if CPE HTTP answers match score signs and policy mapping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from scripts.demo import (
    DEFAULT_THRESHOLD,
    FEATURE_COLS,
    assert_scenario_honesty,
    pick_honest_scenarios,
)
from src.api.main import recommend_treatment
from src.evaluation.metrics import (
    holdout_decision,
    qini_bootstrap_interval,
    qini_random_interval,
    read_ranking_supports_decision,
    split_train_calibration_test,
    uplift_at_k,
)
from src.causal.uplift_models import TLearner
from src.data.synthetic import (
    format_calibration,
    generate_uplift_dataset,
    summarize_calibration,
)
EVIDENCE_PATH = _ROOT / "docs" / "evidence" / "latest_proof.md"


def _wait_health(url: str, timeout: float = 40.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200 and str(r.json().get("model_loaded", "")).lower() == "true":
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _start_cpe() -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8100",
        ],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_health("http://127.0.0.1:8100"):
        proc.terminate()
        raise RuntimeError("CPE failed to become healthy with model loaded")
    return proc


def train_and_pick(threshold: float) -> tuple[dict[str, Any], list, dict[str, Any]]:
    df = generate_uplift_dataset(n=3000, random_state=42)
    treatment = df["treatment"].to_numpy(dtype=int)
    idx_tr, idx_cal, idx_te = split_train_calibration_test(len(df), treatment, random_state=42)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["accepted"].to_numpy(dtype=int)
    X_tr, y_tr, t_tr = X[idx_tr], y[idx_tr], treatment[idx_tr]
    X_cal, y_cal, t_cal = X[idx_cal], y[idx_cal], treatment[idx_cal]
    X_te, y_te, t_te = X[idx_te], y[idx_te], treatment[idx_te]
    model = TLearner(
        base_estimator=GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
    )
    model.fit(X_tr, y_tr, t_tr)
    u_cal = model.predict_uplift(X_cal)
    u_te = model.predict_uplift(X_te)
    cal_qini = qini_bootstrap_interval(y_cal, u_cal, t_cal, n_boot=200, seed=0)
    decision = holdout_decision(float(cal_qini["low"]))
    qini = qini_bootstrap_interval(y_te, u_te, t_te, n_boot=200, seed=1)
    qini_null = qini_random_interval(y_te, t_te, n_draws=200, seed=2)
    segment_fit = summarize_calibration(
        df["segment"].to_numpy()[idx_te],
        df["true_uplift"].to_numpy()[idx_te],
        u_te,
    )
    metrics = {
        "calibration_n": int(len(idx_cal)),
        "calibration_qini": float(cal_qini["point"]),
        "calibration_qini_low": float(cal_qini["low"]),
        "calibration_qini_high": float(cal_qini["high"]),
        "calibration_qini_boot": int(cal_qini["n_boot"]),
        "test_n": int(len(idx_te)),
        "qini_auc": float(qini["point"]),
        "qini_low": float(qini["low"]),
        "qini_high": float(qini["high"]),
        "qini_boot": int(qini["n_boot"]),
        "qini_random_mean": float(qini_null["mean"]),
        "qini_random_low": float(qini_null["low"]),
        "qini_random_high": float(qini_null["high"]),
        "qini_random_draws": int(qini_null["n_draws"]),
        "uplift_at_k": float(uplift_at_k(y_te, u_te, t_te, k=0.3)),
        "calibration": segment_fit,
        **decision,
    }
    scenarios = pick_honest_scenarios(
        X_te,
        u_te,
        FEATURE_COLS,
        threshold=threshold,
        true_uplift=df["true_uplift"].to_numpy()[idx_te],
    )
    assert_scenario_honesty(scenarios, threshold=threshold)

    artifacts = _ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_name": "t_learner",
        "feature_columns": FEATURE_COLS,
        "source": "portfolio_proof_synthetic_hte",
        "metrics": metrics,
        "uplift_threshold": float(threshold),
    }
    joblib.dump(payload, artifacts / "model.joblib")
    return payload, scenarios, metrics


def dpe_policy_from_uplift(
    uplift: float,
    threshold: float,
    *,
    ranking_supports_decision: bool = False,
) -> dict[str, Any]:
    """Mirror the DPE rule for this one treatment.

    A score below -threshold drops the additive surcharge only when the
    holdout Qini interval lies entirely above zero. Otherwise the cut follows
    a ranking that is not separated from noise, and the surcharge stays.
    """
    treatment = recommend_treatment(uplift, threshold=threshold)
    score_says_drop = uplift < -threshold
    override = score_says_drop and ranking_supports_decision
    if override:
        action = "charge the base fare and drop the additive surcharge"
    elif score_says_drop:
        action = (
            "score is below the threshold, but the Qini interval covers 0, "
            "so the surcharge stays"
        )
    elif treatment == "SURCHARGE":
        action = "keep the additive surcharge"
    else:
        action = "keep the quoted surcharge; the score is inside the threshold"
    return {
        "causal_uplift_score": uplift,
        "causal_override": override,
        "ranking_supports_decision": ranking_supports_decision,
        "causal_recommended_treatment": treatment,
        "test_group_if_dpe": "CAUSAL_NO_SURGE" if override else "ADDITIVE",
        "pricing_action": action,
    }


def run_proof(with_dpe: bool = False, threshold: float = DEFAULT_THRESHOLD) -> int:
    print("=" * 64)
    print("  Portfolio Proof — CPE HTTP + DPE policy")
    print("=" * 64)

    print("[1] Train + honest scenario pick (synthetic HTE)...")
    payload, scenarios, metrics = train_and_pick(threshold)
    print(
        f"    Calibration n={metrics['calibration_n']} "
        f"Qini {metrics['calibration_qini']:+.4f} "
        f"95% [{metrics['calibration_qini_low']:+.4f}, {metrics['calibration_qini_high']:+.4f}] "
        f"flag={metrics['ranking_supports_decision']}"
    )
    print(
        f"    Untouched test n={metrics['test_n']} "
        f"Qini {metrics['qini_auc']:+.4f} "
        f"95% [{metrics['qini_low']:+.4f}, {metrics['qini_high']:+.4f}]  "
        f"Uplift@30%={metrics['uplift_at_k']:+.4f}"
    )
    print(format_calibration(metrics["calibration"]))

    cpe_url = "http://127.0.0.1:8100"
    managed: subprocess.Popen | None = None
    evidence_rows: list[dict[str, Any]] = []
    dpe_capture: dict[str, Any] | None = None

    try:
        if not _wait_health(cpe_url, timeout=1.5):
            print("[2] Starting CPE on :8100 ...")
            managed = _start_cpe()
        else:
            print("[2] Reusing already-running CPE on :8100")

        health = httpx.get(f"{cpe_url}/health", timeout=5.0).json()
        print(f"    health: model_loaded={health.get('model_loaded')} name={health.get('model_name')}")

        print("[3] POST /predict_uplift for each honest role...")
        for sc in scenarios:
            body = {"driver_id": f"proof_{sc.role}", "features": sc.features}
            resp = httpx.post(f"{cpe_url}/predict_uplift", json=body, timeout=5.0)
            if resp.status_code != 200:
                print(f"[FAIL] {sc.role}: HTTP {resp.status_code} {resp.text}")
                return 1
            data = resp.json()
            score = float(data["uplift_score"])
            treatment = data["recommended_treatment"]
            expected = recommend_treatment(sc.uplift, threshold=threshold)
            if treatment != expected:
                print(f"[FAIL] {sc.role}: HTTP {treatment} != offline {expected}")
                return 1

            # HTTP score should agree in sign band with offline pick
            if sc.role == "persuadable" and not (score > threshold and treatment == "SURCHARGE"):
                print(f"[FAIL] persuadable HTTP score={score} treatment={treatment}")
                return 1
            if sc.role == "sleeping_dog" and not (
                score < -threshold and treatment == "NO_SURCHARGE"
            ):
                print(f"[FAIL] sleeping_dog HTTP score={score} treatment={treatment}")
                return 1
            if sc.role == "neutral" and treatment not in {"KEEP_QUOTE", "SURCHARGE", "NO_SURCHARGE"}:
                print(f"[FAIL] neutral unexpected treatment={treatment}")
                return 1

            supports = read_ranking_supports_decision(metrics)
            policy = dpe_policy_from_uplift(
                score, threshold, ranking_supports_decision=supports
            )
            print(
                f"    [{sc.role}] τ̂={score:+.4f} → {treatment} | "
                f"DPE override={policy['causal_override']}"
            )
            evidence_rows.append(
                {
                    "role": sc.role,
                    "label": sc.label,
                    "features": sc.features,
                    "offline_uplift": sc.uplift,
                    "planted_uplift": sc.true_uplift,
                    "http_response": data,
                    "dpe_policy_simulation": policy,
                }
            )

        if with_dpe:
            dpe_url = "http://127.0.0.1:8000"
            print("[4] Optional DPE /api/v1/search (best-effort)...")
            try:
                hr = httpx.get(f"{dpe_url}/health", timeout=2.0)
                if hr.status_code != 200:
                    print("    DPE not healthy — skip live search")
                else:
                    sr = httpx.post(
                        f"{dpe_url}/api/v1/search",
                        json={
                            "search_id": "portfolio_proof_1",
                            "driver_id": "proof_driver_1",
                            "lat": 55.7558,
                            "lon": 37.6173,
                            "dest_lat": 55.76,
                            "dest_lon": 37.62,
                        },
                        timeout=10.0,
                    )
                    dpe_capture = {
                        "status_code": sr.status_code,
                        "body": sr.json() if sr.status_code == 200 else sr.text,
                        "note": (
                            "Graph-derived features may differ from synthetic demo rows; "
                            "causal_override is informative, not asserted."
                        ),
                    }
                    if sr.status_code == 200:
                        b = sr.json()
                        print(
                            f"    DPE price={b.get('price')} override={b.get('causal_override')} "
                            f"score={b.get('causal_uplift_score')}"
                        )
            except Exception as exc:
                dpe_capture = {"error": str(exc), "note": "DPE unreachable — CPE proof still valid"}
                print(f"    DPE skip: {exc}")

    finally:
        if managed is not None:
            print("[*] Stopping managed CPE...")
            managed.terminate()
            try:
                managed.wait(timeout=5)
            except Exception:
                managed.kill()

    _write_evidence(evidence_rows, metrics, threshold, health, dpe_capture)
    print(f"\n[OK] Evidence written → {EVIDENCE_PATH}")
    print("=" * 64)
    return 0


def _write_evidence(
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
    threshold: float,
    health: dict[str, Any],
    dpe_capture: dict[str, Any] | None,
) -> None:
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Portfolio proof evidence",
        "",
        f"_Generated automatically by `scripts/portfolio_proof.py` at **{ts}**._",
        "",
        "This file is **captured output**, not hand-written marketing numbers.",
        "",
        "## Setup",
        "",
        f"- Threshold: `±{threshold}`",
        "- Treatment: additive surcharge versus the base fare. Outcome: driver accepts.",
        "- Features: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. "
        "`price` and `surge_bonus` are not features.",
        f"- Calibration slice, n=`{metrics['calibration_n']}`: Qini `{metrics['calibration_qini']:+.4f}`, "
        f"95% bootstrap `[{metrics['calibration_qini_low']:+.4f}, {metrics['calibration_qini_high']:+.4f}]` "
        f"on {metrics['calibration_qini_boot']} resamples. "
        f"This slice alone sets `ranking_supports_decision="
        f"{str(metrics['ranking_supports_decision']).lower()}`.",
        (
            "- The calibration interval contains 0, so the flag is false and DPE does not "
            "drop the surcharge at -0.05. That cut would follow noise."
            if metrics["calibration_qini_low"] < 0 < metrics["calibration_qini_high"]
            else "- The calibration interval does not contain 0."
        ),
        f"- Untouched test, n=`{metrics['test_n']}`: Qini `{metrics['qini_auc']:+.4f}`, "
        f"95% bootstrap `[{metrics['qini_low']:+.4f}, {metrics['qini_high']:+.4f}]` "
        f"on {metrics['qini_boot']} resamples. This slice did not set the flag.",
        f"- Random-score Qini on the untouched test: mean `{metrics['qini_random_mean']:+.4f}`, "
        f"95% `[{metrics['qini_random_low']:+.4f}, {metrics['qini_random_high']:+.4f}]` "
        f"over {metrics['qini_random_draws']} draws. One random draw is not a null.",
        f"- Untouched-test Uplift@30%: `{metrics['uplift_at_k']:+.4f}`",
        f"- CPE `/health`: `{json.dumps(health, ensure_ascii=False)}`",
        "",
        "## Segment means on the untouched test",
        "",
        "The planted cuts sit on columns the model receives. The means below are",
        "on the untouched test. The rows further down are extreme scores on that",
        "same test, so their τ̂ is not the segment mean and is not a success",
        "metric by itself.",
        "",
        "```text",
        format_calibration(metrics["calibration"]),
        "```",
        "",
        "## Extreme scores → live CPE → DPE policy",
        "",
        "DPE asks CPE only on an additive hour that names a driver. "
        "It drops the surcharge only when the calibration flag is true and "
        "`uplift_score < -threshold`. The rows here are the untouched test.",
        "",
    ]
    for row in rows:
        lines.append(f"### `{row['role']}` — {row['label']}")
        lines.append("")
        planted = row.get("planted_uplift")
        planted_text = "n/a" if planted is None else f"{planted:+.2f}"
        lines.append(
            f"- Offline τ̂ (extreme untouched-test row): `{row['offline_uplift']:+.4f}`. "
            f"Planted effect on this same row: `{planted_text}`."
        )
        if planted is not None and abs(float(planted)) > 1e-9:
            ratio = float(row["offline_uplift"]) / float(planted)
            if abs(ratio) >= 1.5:
                lines.append(
                    f"- This row's score is {ratio:.1f} times the planted effect on the same row. "
                    "That is miscalibration of an extreme score, not a confirmed effect."
                )
        lines.append("- HTTP `POST /predict_uplift` response:")
        lines.append("```json")
        lines.append(json.dumps(row["http_response"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("- DPE policy simulation from that score:")
        lines.append("```json")
        lines.append(json.dumps(row["dpe_policy_simulation"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("- Features:")
        lines.append("```json")
        lines.append(json.dumps(row["features"], indent=2))
        lines.append("```")
        lines.append("")

    sd = next(r for r in rows if r["role"] == "sleeping_dog")
    planted = sd.get("planted_uplift")
    planted_text = "unknown" if planted is None else f"{float(planted):+.2f}"
    override = sd["dpe_policy_simulation"]["causal_override"]
    score = sd["http_response"]["uplift_score"]
    label = sd["http_response"]["recommended_treatment"]
    if override:
        decision = "DPE would charge the base fare, because the Qini interval is above 0."
    else:
        decision = (
            "DPE keeps the surcharge. The score is past -0.05, but the Qini interval "
            "covers 0, so the threshold would be cutting on noise."
        )
    lines.extend(
        [
            "## What the lowest score does",
            "",
            f"Captured uplift `{score:+.4f}` → `{label}`. {decision}",
            f"The planted effect on that same row is `{planted_text}`. "
            "The row was chosen because its score is the minimum, not because it "
            "belongs to the sleeping-dog cut.",
            "The segment means above are the calibration check. This row is not.",
            "",
        ]
    )

    if dpe_capture is not None:
        lines.append("## Optional live DPE search")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(dpe_capture, indent=2, ensure_ascii=False, default=str))
        lines.append("```")
        lines.append("")

    EVIDENCE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio CPE/DPE proof with evidence file")
    parser.add_argument("--with-dpe", action="store_true", help="Also call live DPE search if up")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    raise SystemExit(run_proof(with_dpe=args.with_dpe, threshold=args.threshold))


if __name__ == "__main__":
    main()
