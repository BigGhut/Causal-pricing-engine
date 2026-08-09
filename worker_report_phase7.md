# Worker Report Phase 7: Honesty Hardening

## Overview & Execution Summary

All planned tasks from Stages A through E in `handoff_phase7_honesty.md` have been executed, verified, and locally committed without breaking any existing integration contracts or test suites.

---

## Stage-by-Stage Status Checklist

| Stage | Task | Status | Result / Details |
|:---|:---|:---:|:---|
| **A1** | Refresh evidence via `portfolio_proof.py` | DONE | `docs/evidence/latest_proof.md` updated with normalized Qini (`+0.0838`), random null (`+0.0092`), and captured HTTP responses (`persuadable` -> `DISCOUNT_10_PCT`, `neutral` -> `NO_DISCOUNT`, `sleeping_dog` -> `NO_DISCOUNT_AVOID` + DPE `causal_override=true`). |
| **A2** | Disclaimers in `CASE_STUDY.md` & `README.md` | DONE | Explicit caveats added regarding synthetic seeded HTE, normalized Qini metric scale, and DPE observational post-treatment feature entanglement. |
| **A3** | Consistent metric terminology | DONE | Aligned `portfolio_proof.py` and `demo.py` to consistently report normalized Qini coefficient (`+0.0838`) alongside random baseline null (`+0.0092`). |
| **B1** | Holdout evaluation in unit tests | DONE | Updated `test_meta_learners_detect_heterogeneity` in `tests/test_uplift.py` to evaluate on holdout set (`X_te`, `seg_te`) rather than same-X training data. |
| **B2** | Add `test_qini_beats_random_on_holdout` | DONE | Added dedicated test asserting `qini_model > qini_random + 0.02` on synthetic holdout set (`n=1500`, 30% split). Passed cleanly. |
| **B3** | Regression check | DONE | Pytest suite remains clean and fast (39 passed in CPE, 17 passed in DPE). |
| **C1** | `CAUSAL.md` training caveats | DONE | Added "Training Features Caveats" section to `dynamic-pricing-engine/CAUSAL.md` detailing post-treatment feature entanglement (`price`/`surge_bonus`) and pre-treatment feature alternatives. |
| **C2** | Pre-treatment feature mode | DONE | Defined `DPE_PRE_TREATMENT_FEATURES` (`distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`) and updated `load_dpe_data(feature_mode="pre_treatment")` in `src/data/dpe_connector.py`. |
| **C3** | Train CLI integration | DONE | Added `--feature-mode {serve_parity,pre_treatment}` CLI flag to `scripts/train.py` (Default: `serve_parity` for production serve alignment; `pre_treatment` for research/evaluation). |
| **C4** | Connector test coverage | DONE | Added `test_load_dpe_data_feature_modes` in `tests/test_dpe_connector.py`. All tests pass. |
| **D1** | Leakage diagnostics script | DONE | Created `scripts/diagnostics_leakage.py`. Reports propensity AUC, Qini breakdown, and DPE mean surge bonus by arm. Exits 0. |
| **E1** | Local Git commits | DONE | Created clean local commits in both CPE and DPE repos (no remote push). |
| **F/R2**| Online DPE pre-treatment alignment | SKIPPED | Followed Orchestrator Variant R1 recommendation: `pre_treatment` is explicit research/eval mode, `serve_parity` remains online production default. |

---

## Raw Execution Logs & Verification Outputs

### 1. Pytest CPE (`causal-pricing-engine`)
```text
39 passed, 1 warning in 5.05s
```

### 2. Pytest DPE (`dynamic-pricing-engine`)
```text
17 passed, 1 warning in 3.51s
```

### 3. Demo Output (`python scripts/demo.py`)
```text
================================================================
  Causal Pricing Engine — Honest Portfolio Demo
================================================================
Scenarios are real validation rows ranked by predicted τ(x),
not hand-picked feature vectors with decorative labels.

[*] Data: synthetic (n=3000, seeded HTE) | DPE DB available at dpe_database.db (use --source dpe to train on it)
[*] Training T-Learner (GradientBoostingClassifier)...
[*] Holdout Qini coefficient: +0.0838  (normalized; random null +0.0092)
[*] Holdout Uplift@30%:       +0.2302
[*] Decision threshold:       ±0.05
[*] Saved artifact: Z:\pet-project\causal-pricing-engine\artifacts\model.joblib

### Honest uplift decisions (from holdout rows)

| Role | τ̂ (uplift) | past_trips | surge_bonus | Treatment | Policy |
|:---|:---:|:---:|:---:|:---|:---|
| **persuadable** | +0.6899 | 15 | 39.6 | `DISCOUNT_10_PCT` | Act: offer treatment / discount (10%) |
| **neutral** | +0.0002 | 7 | 42.0 | `NO_DISCOUNT` | Act: keep baseline fare (no incremental treatment) |
| **sleeping_dog** | -0.4148 | 16 | 1.4 | `NO_DISCOUNT_AVOID` | Act: suppress surge / avoid treatment (Sleeping Dog) |

Feature vectors (for reproducibility):

  [persuadable] distance_km=14.98, duration_sec=3505.39, price=875.13, surge_bonus=39.60, hour_of_day=0.00, past_trips=15.00, avg_surge=21.22
  [neutral] distance_km=13.68, duration_sec=2842.44, price=776.27, surge_bonus=41.99, hour_of_day=23.00, past_trips=7.00, avg_surge=18.63
  [sleeping_dog] distance_km=14.61, duration_sec=3488.57, price=864.18, surge_bonus=1.40, hour_of_day=4.00, past_trips=16.00, avg_surge=14.60

================================================================
Demo OK — labels match score signs at threshold ±0.05.
================================================================
```

### 4. Portfolio Proof Output (`python scripts/portfolio_proof.py`)
```text
================================================================
  Portfolio Proof — CPE HTTP + DPE policy
================================================================
[1] Train + honest scenario pick (synthetic HTE)...
    Qini coef=+0.0838 (random null +0.0092)  Uplift@30%=+0.2302
[2] Starting CPE on :8100 ...
    health: model_loaded=true name=t_learner
[3] POST /predict_uplift for each honest role...
    [persuadable] τ̂=+0.6899 → DISCOUNT_10_PCT | DPE override=False
    [neutral] τ̂=+0.0002 → NO_DISCOUNT | DPE override=False
    [sleeping_dog] τ̂=-0.4148 → NO_DISCOUNT_AVOID | DPE override=True
[*] Stopping managed CPE...

[OK] Evidence written → Z:\pet-project\causal-pricing-engine\docs\evidence\latest_proof.md
================================================================
```

### 5. Diagnostics Script Output (`python scripts/diagnostics_leakage.py`)
```text
--- [Diagnostics: Synthetic Dataset] ---
Propensity AUC(T|X):        0.5249 (expected ~0.50 for RCT)
Qini Model (normalized):   +0.0838
Qini Random Null:          -0.0064
Qini Oracle:               +1.0000
Uplift@30%:                +0.2302

--- [Diagnostics: DPE Simulation Database] ---
DPE (serve_parity) Propensity AUC(T|X): 0.7828
DPE Mean surge_bonus by arm: T=1 (ADDITIVE): 0.46, T=0 (MULT): 0.00
DPE (pre_treatment) Propensity AUC(T|X): 0.7557
DPE (pre_treatment) Holdout Qini:      +0.1332
```

---

## How to Run `pre_treatment` Mode

For offline research/evaluation on DPE simulation data excluding post-treatment features (`price`, `surge_bonus`):

```bash
cd Z:\pet-project\causal-pricing-engine
python scripts/train.py --source dpe --feature-mode pre_treatment
```

---

## Local Git Commits (No Push)

- **CPE (`causal-pricing-engine`)**: `e85beb5af9ea7e9c923b465f7d2e304bed6617a3`  
  *Commit Message:* `chore: phase7 honesty — normalized Qini, holdout tests, pre_treatment mode, evidence refresh`
- **DPE (`dynamic-pricing-engine`)**: `49dafc4745a08ef2b9ab7fb2dbbbcabb998b57b4`  
  *Commit Message:* `docs: add training features caveats for causal CATE modeling`

---

## Deviations & Unimplemented Items

- **Stage F / R2 (COULD)**: Skipped as planned. Production server path retains 7-feature `serve_parity` contract so DPE integration remains fail-open and fully backward-compatible. `pre_treatment` feature mode is available as a research option for offline training.
