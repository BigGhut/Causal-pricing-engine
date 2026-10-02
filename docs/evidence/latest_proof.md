# Portfolio proof evidence

_Generated automatically by `scripts/portfolio_proof.py` at **2026-10-02 14:46:17 UTC**._

This file is **captured output**, not hand-written marketing numbers.

## Setup

- Threshold: `±0.05`
- Treatment: additive surcharge versus the base fare. Outcome: driver accepts.
- Features: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. `price` and `surge_bonus` are not features.
- Calibration slice, n=`450`: Qini `+0.0569`, 95% bootstrap `[-0.0351, +0.1470]` on 200 resamples. This slice alone sets `ranking_supports_decision=false`.
- The calibration interval contains 0, so the flag is false and DPE does not drop the surcharge at -0.05. That cut would follow noise.
- Untouched test, n=`450`: Qini `+0.0597`, 95% bootstrap `[-0.0204, +0.1508]` on 200 resamples. This slice did not set the flag.
- Random-score Qini on the untouched test: mean `+0.0011`, 95% `[-0.0783, +0.0745]` over 200 draws. One random draw is not a null.
- Untouched-test Uplift@30%: `+0.0774`
- CPE `/health`: `{"status": "ok", "service": "causal-pricing-engine", "model_loaded": "true", "model_name": "t_learner", "source": "portfolio_proof_synthetic_hte", "feature_columns": ["distance_km", "duration_sec", "hour_of_day", "past_trips", "avg_surge"], "ranking_supports_decision": false}`

## Segment means on the untouched test

The planted cuts sit on columns the model receives. The means below are
on the untouched test. The rows further down are extreme scores on that
same test, so their τ̂ is not the segment mean and is not a success
metric by itself.

```text
Planted segments are thresholds on past_trips and distance_km, and both columns are given to the model. persuadable: past_trips >= 8 and distance_km <= 8, tau = +0.25. sleeping_dog: past_trips <= 5 and distance_km >= 9, tau = -0.12. Everyone else has tau = 0. A high score on these columns is not evidence that the model found a hidden group.
  persuadable: n=111 planted +0.25 mean τ̂ +0.188 (predicted/planted = 0.75)
  neutral: n=300 planted +0.00 mean τ̂ -0.019 (no ratio, the planted effect is 0)
  sleeping_dog: n=39 planted -0.12 mean τ̂ -0.104 (predicted/planted = 0.87)
```

## Extreme scores → live CPE → DPE policy

DPE asks CPE only on an additive hour that names a driver. It drops the surcharge only when the calibration flag is true and `uplift_score < -threshold`. The rows here are the untouched test.

### `persuadable` — Highest score — surcharge looks helpful on this row

- Offline τ̂ (extreme untouched-test row): `+0.4056`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_persuadable",
  "uplift_score": 0.40559427161360506,
  "recommended_treatment": "SURCHARGE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.40559427161360506,
  "causal_override": false,
  "ranking_supports_decision": false,
  "causal_recommended_treatment": "SURCHARGE",
  "test_group_if_dpe": "ADDITIVE",
  "pricing_action": "keep the additive surcharge"
}
```
- Features:
```json
{
  "distance_km": 12.347276019237166,
  "duration_sec": 1853.439175540808,
  "hour_of_day": 12.0,
  "past_trips": 9.0,
  "avg_surge": 1.1615232507592532
}
```

### `neutral` — Score nearest zero

- Offline τ̂ (extreme untouched-test row): `-0.0007`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_neutral",
  "uplift_score": -0.0006891968836245099,
  "recommended_treatment": "KEEP_QUOTE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.0006891968836245099,
  "causal_override": false,
  "ranking_supports_decision": false,
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

- Offline τ̂ (extreme untouched-test row): `-0.4208`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_sleeping_dog",
  "uplift_score": -0.42083598211279605,
  "recommended_treatment": "NO_SURCHARGE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.42083598211279605,
  "causal_override": false,
  "ranking_supports_decision": false,
  "causal_recommended_treatment": "NO_SURCHARGE",
  "test_group_if_dpe": "ADDITIVE",
  "pricing_action": "score is below the threshold, but the Qini interval covers 0, so the surcharge stays"
}
```
- Features:
```json
{
  "distance_km": 5.501461880752644,
  "duration_sec": 710.5876118043275,
  "hour_of_day": 0.0,
  "past_trips": 6.0,
  "avg_surge": 28.516564627167654
}
```

## What the lowest score does

Captured uplift `-0.4208` → `NO_SURCHARGE`. DPE keeps the surcharge. The score is past -0.05, but the Qini interval covers 0, so the threshold would be cutting on noise.
The planted effect on that same row is `+0.00`. The row was chosen because its score is the minimum, not because it belongs to the sleeping-dog cut.
The segment means above are the calibration check. This row is not.
