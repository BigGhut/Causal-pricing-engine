"""E2E Smoke Verification Script for CPE and DPE services.

Checks CPE health and predict endpoints, and optionally verifies DPE search
endpoint and fail-open integration behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import httpx


def is_service_healthy(url: str) -> bool:
    """Check if service health endpoint returns 200 OK."""
    try:
        resp = httpx.get(f"{url}/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_service(url: str, timeout: float = 30.0) -> bool:
    """Poll health endpoint until healthy or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_service_healthy(url):
            return True
        time.sleep(0.5)
    return False


def start_server_processes(
    cpe_url: str, dpe_url: str, with_dpe: bool
) -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
    """Optionally spin up background uvicorn servers if not already listening."""
    cpe_proc: subprocess.Popen | None = None
    dpe_proc: subprocess.Popen | None = None

    cpe_dir = Path(__file__).resolve().parent.parent
    dpe_dir = cpe_dir.parent / "dynamic-pricing-engine"

    if not is_service_healthy(cpe_url):
        model_file = cpe_dir / "artifacts" / "model.joblib"
        if not model_file.exists():
            print("[Smoke] Training CPE baseline model first...")
            subprocess.run([sys.executable, "scripts/train.py"], cwd=cpe_dir, check=True)

        print("[Smoke] Starting managed CPE server process on port 8100...")
        cpe_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8100"],
            cwd=cpe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_service(cpe_url, timeout=30.0):
            if cpe_proc:
                cpe_proc.terminate()
            raise RuntimeError(f"Managed CPE server failed to become healthy at {cpe_url}")

    if with_dpe and not is_service_healthy(dpe_url):
        print("[Smoke] Starting managed DPE server process on port 8000...")
        dpe_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=dpe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_service(dpe_url, timeout=30.0):
            if dpe_proc:
                dpe_proc.terminate()
            if cpe_proc:
                cpe_proc.terminate()
            raise RuntimeError(f"Managed DPE server failed to become healthy at {dpe_url}")

    return cpe_proc, dpe_proc


def check_cpe_health(cpe_url: str) -> dict[str, Any]:
    print(f"[Smoke] 1. Checking CPE health at {cpe_url}/health ...")
    resp = httpx.get(f"{cpe_url}/health", timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"CPE health failed with status {resp.status_code}: {resp.text}")

    data = resp.json()
    model_loaded = str(data.get("model_loaded", "")).lower()
    if model_loaded != "true":
        raise RuntimeError(f"CPE model is not loaded! Health response: {data}")

    print(
        f"  [OK] CPE model_loaded=true, model_name='{data.get('model_name')}', "
        f"source='{data.get('source')}', feature_columns={data.get('feature_columns')}"
    )
    return data


def check_cpe_predict(cpe_url: str) -> dict[str, Any]:
    print(f"[Smoke] 2. Testing CPE /predict_uplift ...")
    payload = {
        "driver_id": "smoke_driver_1",
        "features": {
            "distance_km": 6.5,
            "duration_sec": 750.0,
            "hour_of_day": 15.0,
            "past_trips": 4.0,
            "avg_surge": 1.25,
        },
    }
    resp = httpx.post(f"{cpe_url}/predict_uplift", json=payload, timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"CPE predict_uplift failed with status {resp.status_code}: {resp.text}")

    data = resp.json()
    if "uplift_score" not in data or "recommended_treatment" not in data:
        raise RuntimeError(f"Invalid predict response fields: {data}")

    print(
        f"  [OK] CPE predict_uplift score={data['uplift_score']:.4f}, "
        f"treatment='{data['recommended_treatment']}'"
    )
    return data


def check_dpe_integration(dpe_url: str) -> None:
    print(f"[Smoke] 3. Testing DPE integration at {dpe_url} ...")
    # Health check
    h_resp = httpx.get(f"{dpe_url}/health", timeout=5.0)
    if h_resp.status_code != 200:
        raise RuntimeError(f"DPE health failed with status {h_resp.status_code}: {h_resp.text}")

    print(f"  [OK] DPE health ok: {h_resp.json()}")

    # Search request 1 with driver_id
    search_payload_1 = {
        "search_id": "smoke_s_1",
        "driver_id": "smoke_driver_42",
        "lat": 55.7558,
        "lon": 37.6173,
    }
    s_resp1 = httpx.post(f"{dpe_url}/api/v1/search", json=search_payload_1, timeout=5.0)
    if s_resp1.status_code != 200:
        raise RuntimeError(f"DPE search request 1 failed ({s_resp1.status_code}): {s_resp1.text}")

    data1 = s_resp1.json()
    print(
        f"  [OK] DPE search 1 response: price={data1['price']}, "
        f"surge_multiplier={data1['surge_multiplier']}, test_group='{data1['test_group']}', "
        f"causal_score={data1.get('causal_uplift_score')}, causal_override={data1.get('causal_override')}"
    )

    # Search request 2 with same driver_id (tests history accumulation)
    search_payload_2 = {
        "search_id": "smoke_s_2",
        "driver_id": "smoke_driver_42",
        "lat": 55.7558,
        "lon": 37.6173,
    }
    s_resp2 = httpx.post(f"{dpe_url}/api/v1/search", json=search_payload_2, timeout=5.0)
    if s_resp2.status_code != 200:
        raise RuntimeError(f"DPE search request 2 failed ({s_resp2.status_code}): {s_resp2.text}")

    data2 = s_resp2.json()
    print(
        f"  [OK] DPE search 2 response: price={data2['price']}, "
        f"surge_multiplier={data2['surge_multiplier']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end smoke tests for CPE & DPE.")
    parser.add_argument(
        "--cpe-url",
        type=str,
        default="http://localhost:8100",
        help="CPE service base URL (default: http://localhost:8100)",
    )
    parser.add_argument(
        "--dpe-url",
        type=str,
        default="http://localhost:8000",
        help="DPE service base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--with-dpe",
        action="store_true",
        help="Include DPE search endpoint checks in smoke execution",
    )
    parser.add_argument(
        "--start-servers",
        action="store_true",
        help="Automatically spin up background uvicorn server processes if not running",
    )
    args = parser.parse_args()

    print(f"=== E2E Smoke Verification ===")
    cpe_proc, dpe_proc = None, None
    try:
        if args.start_servers:
            cpe_proc, dpe_proc = start_server_processes(args.cpe_url, args.dpe_url, args.with_dpe)

        check_cpe_health(args.cpe_url)
        check_cpe_predict(args.cpe_url)
        if args.with_dpe:
            check_dpe_integration(args.dpe_url)
        print("\n=== Smoke Tests PASSED SUCCESSFULLY ===")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if cpe_proc is not None:
            print("[Smoke] Shutting down managed CPE background process...")
            cpe_proc.terminate()
            try:
                cpe_proc.wait(timeout=5.0)
            except Exception:
                cpe_proc.kill()
        if dpe_proc is not None:
            print("[Smoke] Shutting down managed DPE background process...")
            dpe_proc.terminate()
            try:
                dpe_proc.wait(timeout=5.0)
            except Exception:
                dpe_proc.kill()


if __name__ == "__main__":
    main()

