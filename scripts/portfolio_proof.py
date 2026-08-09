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
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from scripts.demo import (
    DEFAULT_THRESHOLD,
    FEATURE_COLS,
    assert_scenario_honesty,
    pick_honest_scenarios,
)
from src.api.main import recommend_treatment
from src.causal.uplift_models import TLearner
from src.data.synthetic import generate_uplift_dataset
from src.evaluation.metrics import qini_auc_score, uplift_at_k

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


def train_and_pick(threshold: float) -> tuple[dict[str, Any], list, dict[str, float]]:
    df = generate_uplift_dataset(n=3000, random_state=42)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["conversion"].to_numpy(dtype=int)
    t = df["treatment"].to_numpy(dtype=int)
    X_tr, X_va, y_tr, y_va, t_tr, t_va = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=t
    )
    model = TLearner(
        base_estimator=GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
    )
    model.fit(X_tr, y_tr, t_tr)
    u_va = model.predict_uplift(X_va)
    rng = np.random.default_rng(0)
    q_null = float(qini_auc_score(y_va, rng.normal(size=len(y_va)), t_va))
    metrics = {
        "qini_auc": float(qini_auc_score(y_va, u_va, t_va)),
        "qini_null": q_null,
        "uplift_at_k": float(uplift_at_k(y_va, u_va, t_va, k=0.3)),
    }
    scenarios = pick_honest_scenarios(X_va, u_va, FEATURE_COLS, threshold=threshold)
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


def dpe_policy_from_uplift(uplift: float, threshold: float) -> dict[str, Any]:
    """Mirror DPE Sleeping Dog rule (src/api/main.py causal block)."""
    treatment, discount = recommend_treatment(uplift, threshold=threshold)
    override = uplift < -threshold
    return {
        "causal_uplift_score": uplift,
        "causal_override": override,
        "causal_recommended_treatment": treatment,
        "optimal_discount_pct": discount,
        "test_group_if_dpe": "CAUSAL_NO_SURGE" if override else "(unchanged switchback arm)",
        "pricing_action": (
            "force base fare / zero surge_bonus"
            if override
            else (
                "allow treatment / discount path"
                if treatment == "DISCOUNT_10_PCT"
                else "keep rule-based surge"
            )
        ),
    }


def run_proof(with_dpe: bool = False, threshold: float = DEFAULT_THRESHOLD) -> int:
    print("=" * 64)
    print("  Portfolio Proof — CPE HTTP + DPE policy")
    print("=" * 64)

    print("[1] Train + honest scenario pick (synthetic HTE)...")
    payload, scenarios, metrics = train_and_pick(threshold)
    print(
        f"    Qini coef={metrics['qini_auc']:+.4f} (random null {metrics['qini_null']:+.4f})  "
        f"Uplift@30%={metrics['uplift_at_k']:+.4f}"
    )

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
            body = {"user_id": f"proof_{sc.role}", "features": sc.features}
            resp = httpx.post(f"{cpe_url}/predict_uplift", json=body, timeout=5.0)
            if resp.status_code != 200:
                print(f"[FAIL] {sc.role}: HTTP {resp.status_code} {resp.text}")
                return 1
            data = resp.json()
            score = float(data["uplift_score"])
            treatment = data["recommended_treatment"]
            expected, _ = recommend_treatment(sc.uplift, threshold=threshold)

            # HTTP score should agree in sign band with offline pick
            if sc.role == "persuadable" and not (score > threshold and treatment == "DISCOUNT_10_PCT"):
                print(f"[FAIL] persuadable HTTP score={score} treatment={treatment}")
                return 1
            if sc.role == "sleeping_dog" and not (
                score < -threshold and treatment == "NO_DISCOUNT_AVOID"
            ):
                print(f"[FAIL] sleeping_dog HTTP score={score} treatment={treatment}")
                return 1
            if sc.role == "neutral" and treatment not in {"NO_DISCOUNT", "DISCOUNT_10_PCT", "NO_DISCOUNT_AVOID"}:
                print(f"[FAIL] neutral unexpected treatment={treatment}")
                return 1

            policy = dpe_policy_from_uplift(score, threshold)
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
        f"- Holdout Qini coefficient (normalized): `{metrics['qini_auc']:+.4f}` (random null `{metrics['qini_null']:+.4f}`)",
        f"- Holdout Uplift@30%: `{metrics['uplift_at_k']:+.4f}`",
        f"- CPE `/health`: `{json.dumps(health, ensure_ascii=False)}`",
        "",
        "## Honest roles → live CPE → DPE policy rule",
        "",
        "DPE policy (same as production code): "
        "`if uplift_score < -threshold → causal_override, base fare, CAUSAL_NO_SURGE`.",
        "",
    ]
    for row in rows:
        lines.append(f"### `{row['role']}` — {row['label']}")
        lines.append("")
        lines.append(f"- Offline τ̂ (holdout pick): `{row['offline_uplift']:+.4f}`")
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
    assert sd["dpe_policy_simulation"]["causal_override"] is True
    lines.extend(
        [
            "## Sleeping Dog takeaway",
            "",
            f"Captured uplift `{sd['http_response']['uplift_score']:+.4f}` "
            f"→ treatment `{sd['http_response']['recommended_treatment']}` "
            f"→ **causal_override = true** (surge suppressed in DPE policy).",
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
