# Portfolio proof evidence

_Generated automatically by `scripts/portfolio_proof.py` at **2026-10-02 12:51:37 UTC**._

This file is **captured output**, not hand-written marketing numbers.

## Setup

- Threshold: `±0.05`
- Treatment: additive surcharge versus the base fare. Outcome: driver accepts.
- Features: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. `price` and `surge_bonus` are not features.
- Holdout Qini: `+0.0601`, 95% bootstrap `[-0.0026, +0.1163]` on 200 resamples of this one split. The interval contains 0, so the point estimate on this split is not separated from noise.
- Random-score Qini: mean `-0.0020`, 95% `[-0.0612, +0.0587]` over 200 draws. One random draw is not a null.
- Holdout Uplift@30%: `+0.1302`
- CPE `/health`: `{"status": "ok", "service": "causal-pricing-engine", "model_loaded": "true", "model_name": "t_learner", "source": "portfolio_proof_synthetic_hte", "feature_columns": ["distance_km", "duration_sec", "hour_of_day", "past_trips", "avg_surge"]}`

## Calibration

The planted cuts sit on columns the model receives. The mean score in a
segment is the calibration check. The rows below are the extreme scores
on the holdout, so their τ̂ is not the segment mean and is not a success
metric by itself.

```text
Planted segments are thresholds on past_trips and distance_km, and both columns are given to the model. persuadable: past_trips >= 8 and distance_km <= 8, tau = +0.25. sleeping_dog: past_trips <= 5 and distance_km >= 9, tau = -0.12. Everyone else has tau = 0. A high score on these columns is not evidence that the model found a hidden group.
  persuadable: n=224 planted +0.25 mean τ̂ +0.209 (predicted/planted = 0.84)
  neutral: n=602 planted +0.00 mean τ̂ -0.010 (no ratio, the planted effect is 0)
  sleeping_dog: n=74 planted -0.12 mean τ̂ -0.097 (predicted/planted = 0.81)
```

## Extreme scores → live CPE → DPE policy

DPE applies this only on an additive hour that names a driver: `if uplift_score < -threshold → base fare, CAUSAL_NO_SURGE`.

### `persuadable` — Highest score — surcharge looks helpful on this row

- Offline τ̂ (extreme holdout row): `+0.4407`. Planted effect on this same row: `+0.25`.
- This row's score is 1.8 times the planted effect on the same row. That is miscalibration of an extreme score, not a confirmed effect.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_persuadable",
  "uplift_score": 0.4406972834894371,
  "recommended_treatment": "SURCHARGE",
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.4406972834894371,
  "causal_override": false,
  "causal_recommended_treatment": "SURCHARGE",
  "test_group_if_dpe": "ADDITIVE",
  "pricing_action": "keep the additive surcharge"
}
```
- Features:
```json
{
  "distance_km": 1.5538429174626853,
  "duration_sec": 329.4212098682089,
  "hour_of_day": 14.0,
  "past_trips": 11.0,
  "avg_surge": 29.816816655310664
}
```

### `neutral` — Score nearest zero

- Offline τ̂ (extreme holdout row): `-0.0007`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_neutral",
  "uplift_score": -0.0006891968836245099,
  "recommended_treatment": "KEEP_QUOTE",
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.0006891968836245099,
  "causal_override": false,
  "causal_recommended_treatment": "KEEP_QUOTE",
  "test_group_if_dpe": "ADDITIVE",
  "pricing_action": "keep the quoted surcharge; the score is inside the threshold"
}
```
- Features:
```json
{
  "distance_km": 12.375228763471506,
  "duration_sec": 2265.2414696222104,
  "hour_of_day": 14.0,
  "past_trips": 7.0,
  "avg_surge": 19.828185865822615
}
```

### `sleeping_dog` — Lowest score — surcharge looks harmful on this row

- Offline τ̂ (extreme holdout row): `-0.4943`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_sleeping_dog",
  "uplift_score": -0.49434654306435794,
  "recommended_treatment": "NO_SURCHARGE",
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.49434654306435794,
  "causal_override": true,
  "causal_recommended_treatment": "NO_SURCHARGE",
  "test_group_if_dpe": "CAUSAL_NO_SURGE",
  "pricing_action": "charge the base fare and drop the additive surcharge"
}
```
- Features:
```json
{
  "distance_km": 12.97898970383365,
  "duration_sec": 1666.9868696791846,
  "hour_of_day": 11.0,
  "past_trips": 6.0,
  "avg_surge": 26.34621289924354
}
```

## What the lowest score does

Captured uplift `-0.4943` → `NO_SURCHARGE` → DPE would charge the base fare.
The planted effect on that same row is `+0.00`. The row was chosen because its score is the minimum, not because it belongs to the sleeping-dog cut. If the planted effect is near zero, the override fires where the surcharge was not harmful.
The segment means above are the calibration check. This row is not.
