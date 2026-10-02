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
from scripts.demo import (
    DEFAULT_THRESHOLD,
    FEATURE_COLS,
    assert_scenario_honesty,
    pick_honest_scenarios,
)
from src.api.main import recommend_treatment
from src.evaluation.metrics import read_ranking_supports_decision
from src.data.synthetic import format_calibration
from src.evaluation.portfolio_eval import run_split
from src.evaluation.protocol import (
    FALSE_OVERRIDE_RATE_MAX,
    GENERATION_SEEDS,
    N,
    PRIMARY_GENERATION_SEED,
    SPLIT_SEED,
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


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):+.{digits}f}"


def _segment_bias(segments: list[dict], name: str) -> float | None:
    for row in segments:
        if row["segment"] == name:
            return float(row["bias"])
    return None


def train_and_pick(
    threshold: float,
    n: int = N,
) -> tuple[dict[str, Any], list, dict[str, Any], list[dict[str, Any]]]:
    """Run every pre-registered seed. The artifact is generation seed 42, T-learner."""
    records: list[dict[str, Any]] = []
    primary: dict[str, Any] | None = None
    for generation_seed in GENERATION_SEEDS:
        for learner_name in ("t_learner", "x_learner"):
            print(f"    generation seed {generation_seed}, {learner_name}, n={n}")
            result = run_split(
                n=n,
                generation_seed=generation_seed,
                learner_name=learner_name,
                threshold=threshold,
            )
            if n == N and (
                result["n_train"],
                result["n_calibration"],
                result["n_test"],
            ) != (21000, 4500, 4500):
                raise RuntimeError(
                    "Split sizes are not 21000/4500/4500: "
                    f"{result['n_train']}/{result['n_calibration']}/{result['n_test']}"
                )
            keep_arrays = (
                generation_seed == PRIMARY_GENERATION_SEED and learner_name == "t_learner"
            )
            if keep_arrays:
                primary = result
            else:
                for key in ("model", "test_features", "test_predicted", "test_truth"):
                    result.pop(key, None)
            records.append(result)
            print(
                f"      calibration Qini {_fmt(result['calibration_qini'])} "
                f"[{_fmt(result['calibration_qini_low'])}, {_fmt(result['calibration_qini_high'])}] "
                f"false_override={_fmt(result['false_override_rate'])} "
                f"flag={result['ranking_supports_decision']}"
            )
            print(
                f"      test Qini {_fmt(result['test_qini'])} "
                f"[{_fmt(result['test_qini_low'])}, {_fmt(result['test_qini_high'])}] "
                f"PEHE {result['pehe']:.4f}"
            )
    if primary is None:
        raise RuntimeError("pre-registered T-learner seed was not run")

    scenarios = pick_honest_scenarios(
        primary["test_features"],
        primary["test_predicted"],
        FEATURE_COLS,
        threshold=threshold,
        true_uplift=primary["test_truth"],
    )
    assert_scenario_honesty(scenarios, threshold=threshold)
    metrics = {
        "calibration_n": primary["n_calibration"],
        "calibration_qini": primary["calibration_qini"],
        "calibration_qini_low": primary["calibration_qini_low"],
        "calibration_qini_high": primary["calibration_qini_high"],
        "calibration_qini_boot": primary["calibration_n_boot"],
        "test_n": primary["n_test"],
        "qini_auc": primary["test_qini"],
        "qini_low": primary["test_qini_low"],
        "qini_high": primary["test_qini_high"],
        "qini_boot": primary["test_n_boot"],
        "qini_random_mean": primary["random_qini_mean"],
        "qini_random_low": primary["random_qini_low"],
        "qini_random_high": primary["random_qini_high"],
        "qini_random_draws": primary["random_n_draws"],
        "uplift_at_k": primary["uplift_at_k"],
        "calibration": primary["segments"],
        "ranking_supports_decision": primary["ranking_supports_decision"],
        "false_override_rate": primary["false_override_rate"],
        "false_override_rate_max": primary["false_override_rate_max"],
        "false_override_check": primary["false_override_check"],
        "pehe": primary["pehe"],
        "test_false_override_rate": primary["test_false_override_rate"],
        "deciles": primary["deciles"],
        "n": n,
        "split_seed": SPLIT_SEED,
        "generation_seed": PRIMARY_GENERATION_SEED,
    }
    payload = {
        "model": primary["model"],
        "model_name": "t_learner",
        "feature_columns": FEATURE_COLS,
        "source": f"portfolio_proof_seed_{PRIMARY_GENERATION_SEED}_n_{n}",
        "metrics": {
            "ranking_supports_decision": primary["ranking_supports_decision"],
            "false_override_rate": primary["false_override_rate"],
            "false_override_rate_max": FALSE_OVERRIDE_RATE_MAX,
            "false_override_check": primary["false_override_check"],
            "calibration_qini_low": primary["calibration_qini_low"],
            "calibration_qini_high": primary["calibration_qini_high"],
            "qini_auc": primary["test_qini"],
            "qini_low": primary["test_qini_low"],
            "qini_high": primary["test_qini_high"],
        },
        "uplift_threshold": float(threshold),
    }
    artifacts = _ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifacts / "model.joblib")
    for key in ("model", "test_features", "test_predicted", "test_truth"):
        primary.pop(key, None)
    return payload, scenarios, metrics, records


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


def run_proof(
    with_dpe: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    n: int = N,
) -> int:
    print("=" * 64)
    print("  Portfolio Proof — CPE HTTP + DPE policy")
    print("=" * 64)

    print("[1] Pre-registered seeds. The artifact is generation seed 42, T-learner.")
    payload, scenarios, metrics, records = train_and_pick(threshold, n=n)
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
    _write_evidence([], metrics, records, threshold, {}, None)

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

    _write_evidence(evidence_rows, metrics, records, threshold, health, dpe_capture)
    print(f"\n[OK] Evidence written → {EVIDENCE_PATH}")
    print("=" * 64)
    return 0


def _rate_cell(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def _spread(records: list[dict[str, Any]], learner: str, key: str) -> str:
    values = [
        row[key]
        for row in records
        if row["learner"] == learner and row[key] is not None
    ]
    if not values:
        return "undefined"
    numbers = [float(value) for value in values]
    return f"{min(numbers):+.4f} … {max(numbers):+.4f}"


def _tlearner_conclusion(records: list[dict[str, Any]]) -> str:
    t_rows = [row for row in records if row["learner"] == "t_learner"]
    covering = [
        row["generation_seed"]
        for row in t_rows
        if row["test_qini_low"] <= 0.0 <= row["test_qini_high"]
    ]
    if len(covering) == len(t_rows):
        qini_text = (
            "On every generation seed the untouched-test Qini interval for the "
            "T-learner covers 0. At the planted effects +0.25 / −0.12 / 0 this "
            "T-learner is not fit for a surcharge decision."
        )
    elif covering:
        seeds = ", ".join(str(seed) for seed in covering)
        qini_text = (
            f"The untouched-test Qini interval for the T-learner covers 0 on seeds {seeds}. "
            "Those seeds were not dropped. A seed whose interval stays above 0 is not promoted."
        )
    else:
        qini_text = (
            "On these five seeds the untouched-test Qini interval for the T-learner "
            "stays above 0."
        )
    flags_false = t_rows and not any(row["ranking_supports_decision"] for row in t_rows)
    if flags_false:
        return (
            qini_text
            + " The decision flag is false on every T-learner seed in this table. "
            "Where the calibration Qini lower bound is above 0, the false-override "
            "rate is still above the pre-registered maximum 0.10, so the rate check fails. "
            "Under the rule in PLAN.md this T-learner does not change the fare."
        )
    return qini_text


def _write_evidence(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
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
        "Rules and seeds were written in `PLAN.md` on 2026-10-02, before this run.",
        "No seed was dropped, and none was chosen after seeing the metrics.",
        "",
        "## Setup",
        "",
        f"- N: `{metrics['n']}`. Split seed: `{metrics['split_seed']}`. "
        "Shares: 70% train, 15% calibration, 15% untouched test.",
        f"- Generation seeds: `{', '.join(str(seed) for seed in GENERATION_SEEDS)}`.",
        "- Served artifact: T-learner, generation seed "
        f"`{PRIMARY_GENERATION_SEED}`. That seed was named in the plan, not picked as the best.",
        f"- Score threshold: `−{threshold}`. `false_override_rate_max`: `{FALSE_OVERRIDE_RATE_MAX}`.",
        "- A false override is `score < −threshold` on a row with `true_uplift ≥ 0`. "
        "The rate divides by the number of rows where the override fired.",
        "- The flag is true only when the calibration Qini lower bound is above 0 and "
        "the calibration false-override rate is defined and not greater than the maximum.",
        "- Treatment: additive surcharge versus the base fare. Outcome: driver accepts.",
        "- Features: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. "
        "`price` and `surge_bonus` are not features.",
        f"- CPE `/health`: `{json.dumps(health, ensure_ascii=False)}`",
        "",
        "## All five seeds",
        "",
        "| Seed | Learner | Cal Qini | Cal 95% | Cal false rate | Flag | Test Qini | Test 95% | Random 95% | PEHE | Bias +0.25 | Bias 0 | Bias −0.12 | Test false rate |",
        "|:---|:---|---:|:---|---:|:---|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        segments = {row["segment"]: row for row in record["segments"]}
        lines.append(
            "| {seed} | {learner} | {cq:+.4f} | [{cl:+.4f}, {ch:+.4f}] | {crate} | {flag} | "
            "{tq:+.4f} | [{tl:+.4f}, {th:+.4f}] | [{rl:+.4f}, {rh:+.4f}] | {pehe:.4f} | "
            "{bpos} | {bzero} | {bneg} | {trate} |".format(
                seed=record["generation_seed"],
                learner=record["learner"],
                cq=record["calibration_qini"],
                cl=record["calibration_qini_low"],
                ch=record["calibration_qini_high"],
                crate=_rate_cell(record["false_override_rate"]),
                flag=str(record["ranking_supports_decision"]).lower(),
                tq=record["test_qini"],
                tl=record["test_qini_low"],
                th=record["test_qini_high"],
                rl=record["random_qini_low"],
                rh=record["random_qini_high"],
                pehe=record["pehe"],
                bpos=_fmt(_segment_bias(record["segments"], "persuadable")),
                bzero=_fmt(_segment_bias(record["segments"], "neutral")),
                bneg=_fmt(_segment_bias(record["segments"], "sleeping_dog")),
                trate=_rate_cell(record["test_false_override_rate"]),
            )
        )
        del segments
    lines.extend(
        [
            "",
            f"- T-learner test Qini spread: `{_spread(records, 't_learner', 'test_qini')}`.",
            f"- T-learner test PEHE spread: `{_spread(records, 't_learner', 'pehe')}`.",
            f"- X-learner test Qini spread: `{_spread(records, 'x_learner', 'test_qini')}`.",
            f"- X-learner test PEHE spread: `{_spread(records, 'x_learner', 'pehe')}`.",
            "",
            "## Pre-registered seed, untouched test",
            "",
            "Generation seed "
            f"`{metrics['generation_seed']}`, T-learner. "
            f"Calibration n=`{metrics['calibration_n']}`, test n=`{metrics['test_n']}`.",
            f"- Calibration Qini `{metrics['calibration_qini']:+.4f}`, "
            f"95% `[{metrics['calibration_qini_low']:+.4f}, {metrics['calibration_qini_high']:+.4f}]` "
            f"on {metrics['calibration_qini_boot']} resamples. "
            f"False-override rate `{_rate_cell(metrics['false_override_rate'])}` "
            f"against maximum `{metrics['false_override_rate_max']}`. "
            f"Check `{metrics['false_override_check']}`. "
            f"Flag `{str(metrics['ranking_supports_decision']).lower()}`.",
            f"- Untouched-test Qini `{metrics['qini_auc']:+.4f}`, "
            f"95% `[{metrics['qini_low']:+.4f}, {metrics['qini_high']:+.4f}]` "
            f"on {metrics['qini_boot']} resamples. This slice did not set the flag.",
            f"- Random-score Qini on that test: mean `{metrics['qini_random_mean']:+.4f}`, "
            f"95% `[{metrics['qini_random_low']:+.4f}, {metrics['qini_random_high']:+.4f}]` "
            f"over {metrics['qini_random_draws']} draws.",
            f"- PEHE `{metrics['pehe']:.4f}`.",
            f"- Test false-override rate `{_rate_cell(metrics['test_false_override_rate'])}`.",
            f"- Uplift@30% `{metrics['uplift_at_k']:+.4f}`.",
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
    )
    lines.extend(["## Deciles on the untouched test", "",
        "Decile 1 has the lowest predicted scores. Decile 10 has the highest.",
        "The observed difference is mean acceptance among treated rows in the decile minus mean acceptance among control rows.",
        "An empty observed cell means the decile did not contain both arms.",
        ""])
    for record in records:
        lines.append(
            f"### Seed `{record['generation_seed']}`, `{record['learner']}`"
        )
        lines.append("")
        lines.append("| Decile | n | mean τ̂ | observed difference |")
        lines.append("|---:|---:|---:|---:|")
        for decile in record["deciles"]:
            observed = decile["observed_difference"]
            observed_text = "empty" if observed is None else f"{float(observed):+.4f}"
            lines.append(
                f"| {decile['decile']} | {decile['n']} | "
                f"{float(decile['mean_predicted']):+.4f} | {observed_text} |"
            )
        lines.append("")
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

    if not rows:
        lines.extend(["## HTTP rows", "", "Not captured in this draft.", ""])
        lines.extend(["## What the test intervals say", "", _tlearner_conclusion(records), ""])
        EVIDENCE_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    sd = next(r for r in rows if r["role"] == "sleeping_dog")
    planted = sd.get("planted_uplift")
    planted_text = "unknown" if planted is None else f"{float(planted):+.2f}"
    override = sd["dpe_policy_simulation"]["causal_override"]
    score = sd["http_response"]["uplift_score"]
    label = sd["http_response"]["recommended_treatment"]
    if override:
        decision = (
            "DPE would charge the base fare. The calibration flag is true: "
            "the Qini lower bound is above 0 and the false-override rate is within 0.10."
        )
    else:
        decision = (
            "DPE keeps the surcharge. The calibration flag is false, so a score "
            "below -0.05 does not change the fare."
        )
    lines.extend(
        [
            "## What the lowest score does",
            "",
            f"Captured uplift `{score:+.4f}` → `{label}`. {decision}",
            f"The planted effect on that same row is `{planted_text}`. "
            "The row was chosen because its score is the minimum, not because it "
            "belongs to the sleeping-dog cut.",
            "The segment means above are on the untouched test. This row is an extreme score, not the segment mean.",
            "",
            "## What the test intervals say",
            "",
            _tlearner_conclusion(records),
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
    parser.add_argument("--n", type=int, default=N)
    args = parser.parse_args()
    raise SystemExit(run_proof(with_dpe=args.with_dpe, threshold=args.threshold, n=args.n))


if __name__ == "__main__":
    main()
