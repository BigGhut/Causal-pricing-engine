# Portfolio proof evidence

_Generated automatically by `scripts/portfolio_proof.py` at **2026-08-09 11:04:40 UTC**._

This file is **captured output**, not hand-written marketing numbers.

## Setup

- Threshold: `±0.05`
- Holdout Qini coefficient (normalized): `+0.0838` (random null `+0.0092`)
- Holdout Uplift@30%: `+0.2302`
- CPE `/health`: `{"status": "ok", "service": "causal-pricing-engine", "model_loaded": "true", "model_name": "t_learner", "source": "portfolio_proof_synthetic_hte", "feature_columns": ["distance_km", "duration_sec", "price", "surge_bonus", "hour_of_day", "past_trips", "avg_surge"]}`

## Honest roles → live CPE → DPE policy rule

DPE policy (same as production code): `if uplift_score < -threshold → causal_override, base fare, CAUSAL_NO_SURGE`.

### `persuadable` — Persuadable — treatment likely helps

- Offline τ̂ (holdout pick): `+0.6899`
- HTTP `POST /predict_uplift` response:
```json
{
  "user_id": "proof_persuadable",
  "uplift_score": 0.6899172798969054,
  "recommended_treatment": "DISCOUNT_10_PCT",
  "optimal_discount_pct": 10.0,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.6899172798969054,
  "causal_override": false,
  "causal_recommended_treatment": "DISCOUNT_10_PCT",
  "optimal_discount_pct": 10.0,
  "test_group_if_dpe": "(unchanged switchback arm)",
  "pricing_action": "allow treatment / discount path"
}
```
- Features:
```json
{
  "distance_km": 14.98360055336727,
  "duration_sec": 3505.3943574235127,
  "price": 875.129449576533,
  "surge_bonus": 39.603293754564866,
  "hour_of_day": 0.0,
  "past_trips": 15.0,
  "avg_surge": 21.216224549214903
}
```

### `neutral` — Neutral — little incremental effect

- Offline τ̂ (holdout pick): `+0.0002`
- HTTP `POST /predict_uplift` response:
```json
{
  "user_id": "proof_neutral",
  "uplift_score": 0.00024444568827530766,
  "recommended_treatment": "NO_DISCOUNT",
  "optimal_discount_pct": 0.0,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.00024444568827530766,
  "causal_override": false,
  "causal_recommended_treatment": "NO_DISCOUNT",
  "optimal_discount_pct": 0.0,
  "test_group_if_dpe": "(unchanged switchback arm)",
  "pricing_action": "keep rule-based surge"
}
```
- Features:
```json
{
  "distance_km": 13.681085940799642,
  "duration_sec": 2842.444757799054,
  "price": 776.2716242998965,
  "surge_bonus": 41.988062412514026,
  "hour_of_day": 23.0,
  "past_trips": 7.0,
  "avg_surge": 18.632625552618126
}
```

### `sleeping_dog` — Sleeping Dog — treatment likely hurts

- Offline τ̂ (holdout pick): `-0.4148`
- HTTP `POST /predict_uplift` response:
```json
{
  "user_id": "proof_sleeping_dog",
  "uplift_score": -0.41484767524947835,
  "recommended_treatment": "NO_DISCOUNT_AVOID",
  "optimal_discount_pct": 0.0,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.41484767524947835,
  "causal_override": true,
  "causal_recommended_treatment": "NO_DISCOUNT_AVOID",
  "optimal_discount_pct": 0.0,
  "test_group_if_dpe": "CAUSAL_NO_SURGE",
  "pricing_action": "force base fare / zero surge_bonus"
}
```
- Features:
```json
{
  "distance_km": 14.612847471070372,
  "duration_sec": 3488.565829135661,
  "price": 864.1777696903255,
  "surge_bonus": 1.3993279604450182,
  "hour_of_day": 4.0,
  "past_trips": 16.0,
  "avg_surge": 14.603724806006504
}
```

## Sleeping Dog takeaway

Captured uplift `-0.4148` → treatment `NO_DISCOUNT_AVOID` → **causal_override = true** (surge suppressed in DPE policy).
