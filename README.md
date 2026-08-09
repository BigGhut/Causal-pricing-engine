# Causal Pricing Engine (CPE)

Causal Pricing Engine (CPE) is a **Causal ML & Uplift Modeling** microservice designed to evaluate Individual Treatment Effects (ITE) for dynamic pricing and discount allocation. Trained on simulation logs from its companion project, the [Dynamic Pricing Engine (DPE)](../dynamic-pricing-engine), CPE scores customer uplift in real-time to suppress dynamic surge multipliers for sensitive segments ("Sleeping Dogs") while optimizing incremental conversion and driver payouts.

---

## Почему не классический ML?

Классические модели supervised learning оценивают уровень конверсии $P(Y=1 \mid X, T)$. Они не отделяют тех, кто конвертируется **и без** стимула, от тех, чьё поведение **реально меняется** из‑за воздействия (скидка, surge и т.п.).

Uplift-моделирование напрямую оценивает **индивидуальный эффект воздействия (ITE)**:
$$\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]$$

где $Y^{(1)}$ — исход при treatment (например, скидка или surge-надбавка), $Y^{(0)}$ — исход без воздействия (control), а $\tau(x)$ — **чистый прирост**, который можно отнести именно к treatment.

> **Оговорки и границы claim.** Дефолтное демо использует **синтетику с заложенным HTE**: оно показывает механику пайплайна и правила политики, а **не** city-scale lift в проде. Коэффициент Qini — **нормализованный** ($\approx [-1, 1]$, random null $\approx 0$). Логи симуляции DPE — observational / switchback-данные с возможным post-treatment entanglement признаков (см. `CASE_STUDY.md`).

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

# 2. Honest in-process demo (labels match score signs)
python scripts/demo.py

# 3. Portfolio proof: live CPE HTTP + DPE policy rule → docs/evidence/
python scripts/portfolio_proof.py
# make proof

# 4. (Optional) Dual service stack + proof against live DPE
# Terminal 1: uvicorn src.api.main:app --port 8100
# Terminal 2: cd ../dynamic-pricing-engine && uvicorn src.api.main:app --port 8000
# python scripts/portfolio_proof.py --with-dpe
```

Makefile: `make demo`, `make proof`, `make train`, `make test`, `make smoke-local`.

**Read next:** [CASE_STUDY.md](CASE_STUDY.md) (problem → approach → trade-offs) · [docs/evidence/latest_proof.md](docs/evidence/latest_proof.md) (captured JSON after `make proof`).

---

## Honest Demo & Proof

### Demo (`scripts/demo.py`)

Does **not** invent decorative feature vectors. Scores holdout and picks three **real rows**:

| Role | Selection rule | Policy at ±0.05 |
|:---|:---|:---|
| **persuadable** | max τ̂ | apply treatment |
| **neutral** | τ̂ ≈ 0 | baseline fare |
| **sleeping_dog** | min τ̂ | suppress surge |

If both clear +/− HTE cannot be found, the demo **fails** (or retries synthetic) instead of lying.

### Proof (`scripts/portfolio_proof.py`)

1. Same honest picks  
2. Starts CPE if needed  
3. `POST /predict_uplift` for each role (asserts treatment codes)  
4. Applies **the same override rule as DPE** (`τ̂ < −θ → causal_override`)  
5. Writes **captured** output to `docs/evidence/latest_proof.md`  

Optional `--with-dpe` records a live `/api/v1/search` (graph features may differ; override there is informative, not asserted).
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

1. Larger switchback logs + CUPED readout of the override policy (not only Qini on holdout).  
2. Multi-treatment (surge levels / discount depths) instead of binary treat/control.  
3. Shared online feature store so graph features under load match offline training columns.

---

## Process archive

Multi-agent handoffs and audits: [`docs/archive/`](docs/archive/README.md).  
DPE integration notes: [`CAUSAL.md`](../dynamic-pricing-engine/CAUSAL.md).
