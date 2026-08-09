# Causal Pricing Engine (CPE)

Causal Pricing Engine (CPE) is a **Causal ML & Uplift Modeling** microservice designed to evaluate Individual Treatment Effects (ITE) for dynamic pricing and discount allocation. Trained on simulation logs from its companion project, the [Dynamic Pricing Engine (DPE)](../dynamic-pricing-engine), CPE scores customer uplift in real-time to suppress dynamic surge multipliers for sensitive segments ("Sleeping Dogs") while optimizing incremental conversion and driver payouts.

---

## Why Not Classic ML?

Classic supervised machine learning models predict baseline conversion rate $P(Y=1 \mid X, T)$, which fails to separate users who convert regardless of incentives from those whose behavior is genuinely shifted by the intervention.

Uplift modeling directly estimates the Individual Treatment Effect (ITE):
$$\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]$$

Where $Y^{(1)}$ is the outcome under treatment (e.g. discount or surge bonus), $Y^{(0)}$ is the baseline outcome under control, and $\tau(x)$ represents the net incremental gain attributable strictly to the treatment.

---

## System Architecture

```text
                                +-------------------------------+
                                |  Dynamic Pricing Engine (DPE) |
                                |       FastAPI on port :8000   |
                                +---------------+---------------+
                                                |
                                    POST /predict_uplift
                                  (timeout <= 200ms, fail-open)
                                                |
                                                v
+-----------------------+       +---------------+---------------+
|  DriverHistoryStore   | ----> |  Causal Pricing Engine (CPE)  |
| (Train/Serve Parity)  |       |       FastAPI on port :8100   |
+-----------------------+       +---------------+---------------+
                                                |
                                        [T-Learner / DML]
                                                |
                                                v
                                 Returns ITE score & treatment
```

---

## Key Architectural Ideas

- **Persuadables vs. Sleeping Dogs**:
  - *Persuadables ($\tau(x) > 0$)*: Users who convert specifically because of treatment (offered targeted discount/surge bonus).
  - *Sleeping Dogs ($\tau(x) < 0$)*: Sensitive users whose conversion/acceptance drops under surge pricing.
- **Sleeping Dog Override**: When DPE detects a negative uplift score below threshold ($\tau(x) < -0.05$), it automatically overrides the surge pricing to base fare (`CAUSAL_NO_SURGE`), protecting user retention.
- **Fail-Open Resilience**: All calls from DPE to CPE feature strict HTTP timeouts ($\le 200\text{ms}$). If CPE is offline or slow, DPE safely falls back to standard rule-based pricing without blocking requests.
- **Train/Serve Parity**: Features such as `past_trips` and `avg_surge` use strict point-in-time state (as-of prior trip) in both offline dataset generation (`load_dpe_data`) and online feature tracking (`DriverHistoryStore`).

---

## Quick Start

```powershell
# 1. Install lightweight dependencies
pip install -r requirements.txt
pip install -e .

# 2. Run in-process portfolio demo
python scripts/demo.py

# 3. (Optional) Train model on DPE SQLite simulation data
python scripts/train.py --source dpe

# 4. (Optional) Run dual service stack locally
# Terminal 1: uvicorn src.api.main:app --port 8100
# Terminal 2: cd ../dynamic-pricing-engine && uvicorn src.api.main:app --port 8000
```

For Makefile shortcuts, run `make demo`, `make train`, `make test`, or `make smoke-local`.

---

## Model Performance & Inference Output

### Offline Evaluation (Qini Curve AUC)
Models are evaluated on holdout simulation data using the Qini AUC metric:
- **T-Learner (GBM)**: Qini AUC = `551.72` | Uplift@30% = `0.108`
- **Double ML (LinearDML)**: Qini AUC = `598.42` | Uplift@30% = `-0.035`

### Sample Prediction Request (`POST /predict_uplift`)
```json
{
  "user_id": "drv_1042",
  "features": {
    "distance_km": 7.5,
    "duration_sec": 900.0,
    "price": 320.0,
    "surge_bonus": 45.0,
    "hour_of_day": 18.0,
    "past_trips": 15.0,
    "avg_surge": 20.0
  }
}
```

### Sample DPE Response with Causal Fields (`POST /api/v1/search`)
```json
{
  "h3_index": "881180e001fffff",
  "price": 320.0,
  "surge_multiplier": 1.0,
  "explanation": "Graph-based pricing. Causal override (Sleeping Dog): uplift=-0.0840 < -0.05.",
  "causal_uplift_score": -0.0840,
  "causal_override": true,
  "causal_recommended_treatment": "NO_DISCOUNT_AVOID"
}
```

---

## Project Layout

```text
causal-pricing-engine/
├── artifacts/              # Serialized model joblib payloads
├── configs/                # System configuration (config.yaml)
├── docs/archive/           # Historical multi-agent build notes & audits
├── scripts/
│   ├── demo.py             # Self-contained in-process demo
│   ├── train.py            # Model training & candidate selection CLI
│   └── e2e_smoke.py        # End-to-end service integration verifier
├── src/
│   ├── api/main.py         # FastAPI service (port :8100)
│   ├── causal/             # Uplift meta-learners & DML engine
│   ├── data/               # Unified feature contract & DPE SQLite connector
│   └── evaluation/         # Qini curve & Uplift@k metrics
├── tests/                  # Pytest verification suite
├── Makefile                # Command shortcuts
└── pyproject.toml          # Package configuration & optional deps
```

---

## What I'd Improve Next

1. **Continuous Policy Optimization**: Replace static thresholding with dynamic contextual bandit / offline reinforcement learning (e.g. Doubly Robust Policy Evaluation).
2. **Feature Store Synchronization**: Replace direct SQLite queries with real-time Redis feature streaming for sub-millisecond feature lookup.
3. **Adaptive Switchback Windowing**: Dynamically resize spatial-temporal switchback blocks based on real-time graph congestion metrics.

---

## Process Artifacts & Multi-Agent Build History

All phase handoff notes, audit reports, and multi-agent development logs are preserved in [`docs/archive/`](docs/archive/README.md).
For more details on DPE integration, see [`CAUSAL.md`](../dynamic-pricing-engine/CAUSAL.md) in the DPE repository.
