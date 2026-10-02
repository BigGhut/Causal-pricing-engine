# Portfolio proof evidence

_Generated automatically by `scripts/portfolio_proof.py` at **2026-10-02 15:04:27 UTC**._

This file is **captured output**, not hand-written marketing numbers.
Rules and seeds were written in `PLAN.md` on 2026-10-02, before this run.
No seed was dropped, and none was chosen after seeing the metrics.

## Setup

- N: `30000`. Split seed: `20261002`. Shares: 70% train, 15% calibration, 15% untouched test.
- Generation seeds: `42, 7, 11, 99, 123`.
- Served artifact: T-learner, generation seed `42`. That seed was named in the plan, not picked as the best.
- Score threshold: `−0.05`. `false_override_rate_max`: `0.1`.
- A false override is `score < −threshold` on a row with `true_uplift ≥ 0`. The rate divides by the number of rows where the override fired.
- The flag is true only when the calibration Qini lower bound is above 0 and the calibration false-override rate is defined and not greater than the maximum.
- Treatment: additive surcharge versus the base fare. Outcome: driver accepts.
- Features: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. `price` and `surge_bonus` are not features.
- CPE `/health`: `{"status": "ok", "service": "causal-pricing-engine", "model_loaded": "true", "model_name": "t_learner", "source": "portfolio_proof_seed_42_n_30000", "feature_columns": ["distance_km", "duration_sec", "hour_of_day", "past_trips", "avg_surge"], "ranking_supports_decision": false}`

## All five seeds

| Seed | Learner | Cal Qini | Cal 95% | Cal false rate | Flag | Test Qini | Test 95% | Random 95% | PEHE | Bias +0.25 | Bias 0 | Bias −0.12 | Test false rate |
|:---|:---|---:|:---|---:|:---|---:|:---|:---|---:|---:|---:|---:|---:|
| 42 | t_learner | +0.0993 | [+0.0705, +0.1257] | 0.3260 | false | +0.0985 | [+0.0716, +0.1253] | [-0.0267, +0.0279] | 0.0512 | -0.0032 | +0.0147 | +0.0225 | 0.3365 |
| 42 | x_learner | +0.0966 | [+0.0696, +0.1256] | 0.3190 | false | +0.0838 | [+0.0566, +0.1109] | [-0.0267, +0.0279] | 0.0818 | -0.0741 | +0.0465 | +0.0278 | 0.3386 |
| 7 | t_learner | +0.1132 | [+0.0854, +0.1405] | 0.5932 | false | +0.0946 | [+0.0668, +0.1221] | [-0.0272, +0.0271] | 0.0531 | -0.0125 | -0.0080 | +0.0090 | 0.5793 |
| 7 | x_learner | +0.0990 | [+0.0716, +0.1266] | 0.5551 | false | +0.1037 | [+0.0779, +0.1301] | [-0.0272, +0.0271] | 0.0797 | -0.0889 | +0.0227 | +0.0032 | 0.5591 |
| 11 | t_learner | +0.0954 | [+0.0663, +0.1222] | 0.2842 | false | +0.0970 | [+0.0691, +0.1238] | [-0.0278, +0.0288] | 0.0487 | -0.0118 | +0.0208 | +0.0329 | 0.2851 |
| 11 | x_learner | +0.0650 | [+0.0383, +0.0922] | 0.3661 | false | +0.0927 | [+0.0637, +0.1181] | [-0.0278, +0.0288] | 0.0807 | -0.0809 | +0.0475 | +0.0488 | 0.3325 |
| 99 | t_learner | +0.0993 | [+0.0731, +0.1275] | 0.5614 | false | +0.0917 | [+0.0630, +0.1183] | [-0.0263, +0.0256] | 0.0501 | -0.0072 | -0.0073 | +0.0038 | 0.5617 |
| 99 | x_learner | +0.0907 | [+0.0620, +0.1190] | 0.5000 | false | +0.0890 | [+0.0618, +0.1170] | [-0.0263, +0.0256] | 0.0796 | -0.0931 | +0.0243 | +0.0078 | 0.5159 |
| 123 | t_learner | +0.0841 | [+0.0561, +0.1106] | 0.4897 | false | +0.0915 | [+0.0634, +0.1172] | [-0.0273, +0.0271] | 0.0489 | +0.0029 | +0.0030 | +0.0303 | 0.5166 |
| 123 | x_learner | +0.0918 | [+0.0638, +0.1182] | 0.3845 | false | +0.0890 | [+0.0615, +0.1147] | [-0.0273, +0.0271] | 0.0788 | -0.0827 | +0.0408 | +0.0319 | 0.3745 |

- T-learner test Qini spread: `+0.0915 … +0.0985`.
- T-learner test PEHE spread: `+0.0487 … +0.0531`.
- X-learner test Qini spread: `+0.0838 … +0.1037`.
- X-learner test PEHE spread: `+0.0788 … +0.0818`.

## Pre-registered seed, untouched test

Generation seed `42`, T-learner. Calibration n=`4500`, test n=`4500`.
- Calibration Qini `+0.0993`, 95% `[+0.0705, +0.1257]` on 1000 resamples. False-override rate `0.3260` against maximum `0.1`. Check `fail`. Flag `false`.
- Untouched-test Qini `+0.0985`, 95% `[+0.0716, +0.1253]` on 1000 resamples. This slice did not set the flag.
- Random-score Qini on that test: mean `+0.0002`, 95% `[-0.0267, +0.0279]` over 1000 draws.
- PEHE `0.0512`.
- Test false-override rate `0.3365`.
- Uplift@30% `+0.1911`.

## Segment means on the untouched test

The planted cuts sit on columns the model receives. The means below are
on the untouched test. The rows further down are extreme scores on that
same test, so their τ̂ is not the segment mean and is not a success
metric by itself.

```text
Planted segments are thresholds on past_trips and distance_km, and both columns are given to the model. persuadable: past_trips >= 8 and distance_km <= 8, tau = +0.25. sleeping_dog: past_trips <= 5 and distance_km >= 9, tau = -0.12. Everyone else has tau = 0. A high score on these columns is not evidence that the model found a hidden group.
  persuadable: n=1203 planted +0.25 mean τ̂ +0.247 (predicted/planted = 0.99)
  neutral: n=2904 planted +0.00 mean τ̂ +0.015 (no ratio, the planted effect is 0)
  sleeping_dog: n=393 planted -0.12 mean τ̂ -0.098 (predicted/planted = 0.81)
```

## Extreme scores → live CPE → DPE policy

DPE asks CPE only on an additive hour that names a driver. It drops the surcharge only when the calibration flag is true and `uplift_score < -threshold`. The rows here are the untouched test.

## Deciles on the untouched test

Decile 1 has the lowest predicted scores. Decile 10 has the highest.
The observed difference is mean acceptance among treated rows in the decile minus mean acceptance among control rows.
An empty observed cell means the decile did not contain both arms.

### Seed `42`, `t_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1067 | -0.1330 |
| 2 | 450 | -0.0365 | -0.0222 |
| 3 | 450 | -0.0112 | -0.0091 |
| 4 | 450 | +0.0037 | +0.0115 |
| 5 | 450 | +0.0169 | +0.0419 |
| 6 | 450 | +0.0328 | +0.0489 |
| 7 | 450 | +0.0638 | +0.0629 |
| 8 | 450 | +0.1708 | +0.1755 |
| 9 | 450 | +0.2403 | +0.1556 |
| 10 | 450 | +0.2954 | +0.2664 |

### Seed `42`, `x_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.0946 | -0.1272 |
| 2 | 450 | -0.0358 | -0.0548 |
| 3 | 450 | +0.0010 | +0.0994 |
| 4 | 450 | +0.0295 | +0.0030 |
| 5 | 450 | +0.0548 | +0.0638 |
| 6 | 450 | +0.0801 | +0.0591 |
| 7 | 450 | +0.1066 | +0.0569 |
| 8 | 450 | +0.1345 | +0.1160 |
| 9 | 450 | +0.1704 | +0.2052 |
| 10 | 450 | +0.2432 | +0.1509 |

### Seed `7`, `t_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1246 | -0.0864 |
| 2 | 450 | -0.0599 | -0.0073 |
| 3 | 450 | -0.0323 | -0.0129 |
| 4 | 450 | -0.0175 | -0.0616 |
| 5 | 450 | -0.0040 | +0.0002 |
| 6 | 450 | +0.0123 | +0.0755 |
| 7 | 450 | +0.0421 | -0.0029 |
| 8 | 450 | +0.1544 | +0.1550 |
| 9 | 450 | +0.2331 | +0.1689 |
| 10 | 450 | +0.2874 | +0.2566 |

### Seed `7`, `x_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1208 | -0.1009 |
| 2 | 450 | -0.0601 | -0.0800 |
| 3 | 450 | -0.0248 | -0.0668 |
| 4 | 450 | +0.0068 | -0.0316 |
| 5 | 450 | +0.0353 | +0.1158 |
| 6 | 450 | +0.0605 | +0.0506 |
| 7 | 450 | +0.0881 | +0.0667 |
| 8 | 450 | +0.1198 | +0.0952 |
| 9 | 450 | +0.1560 | +0.1492 |
| 10 | 450 | +0.2217 | +0.2652 |

### Seed `11`, `t_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.0894 | -0.1279 |
| 2 | 450 | -0.0293 | +0.0667 |
| 3 | 450 | -0.0015 | -0.0652 |
| 4 | 450 | +0.0139 | +0.0240 |
| 5 | 450 | +0.0261 | +0.0065 |
| 6 | 450 | +0.0396 | -0.0250 |
| 7 | 450 | +0.0669 | +0.0133 |
| 8 | 450 | +0.1813 | +0.2205 |
| 9 | 450 | +0.2364 | +0.2059 |
| 10 | 450 | +0.2803 | +0.2483 |

### Seed `11`, `x_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.0774 | -0.0445 |
| 2 | 450 | -0.0261 | -0.0702 |
| 3 | 450 | +0.0066 | +0.0289 |
| 4 | 450 | +0.0335 | -0.0112 |
| 5 | 450 | +0.0592 | -0.0621 |
| 6 | 450 | +0.0840 | +0.0476 |
| 7 | 450 | +0.1087 | +0.0619 |
| 8 | 450 | +0.1355 | +0.1190 |
| 9 | 450 | +0.1668 | +0.2247 |
| 10 | 450 | +0.2243 | +0.2505 |

### Seed `99`, `t_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1319 | -0.0380 |
| 2 | 450 | -0.0623 | +0.0132 |
| 3 | 450 | -0.0330 | -0.0097 |
| 4 | 450 | -0.0129 | +0.0578 |
| 5 | 450 | +0.0032 | -0.0587 |
| 6 | 450 | +0.0182 | +0.0361 |
| 7 | 450 | +0.0400 | +0.0548 |
| 8 | 450 | +0.1385 | +0.1156 |
| 9 | 450 | +0.2446 | +0.2531 |
| 10 | 450 | +0.2865 | +0.2848 |

### Seed `99`, `x_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1173 | -0.0222 |
| 2 | 450 | -0.0581 | -0.0433 |
| 3 | 450 | -0.0224 | +0.0639 |
| 4 | 450 | +0.0066 | -0.0197 |
| 5 | 450 | +0.0353 | -0.0317 |
| 6 | 450 | +0.0614 | +0.1108 |
| 7 | 450 | +0.0857 | +0.0286 |
| 8 | 450 | +0.1145 | +0.1805 |
| 9 | 450 | +0.1503 | +0.1443 |
| 10 | 450 | +0.2174 | +0.3022 |

### Seed `123`, `t_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.1018 | -0.0509 |
| 2 | 450 | -0.0473 | -0.0466 |
| 3 | 450 | -0.0194 | -0.0285 |
| 4 | 450 | -0.0035 | +0.0331 |
| 5 | 450 | +0.0096 | -0.0097 |
| 6 | 450 | +0.0261 | +0.0420 |
| 7 | 450 | +0.0528 | +0.0164 |
| 8 | 450 | +0.1657 | +0.1789 |
| 9 | 450 | +0.2458 | +0.1516 |
| 10 | 450 | +0.3034 | +0.2909 |

### Seed `123`, `x_learner`

| Decile | n | mean τ̂ | observed difference |
|---:|---:|---:|---:|
| 1 | 450 | -0.0858 | -0.0867 |
| 2 | 450 | -0.0312 | -0.0837 |
| 3 | 450 | +0.0012 | +0.0096 |
| 4 | 450 | +0.0275 | +0.0385 |
| 5 | 450 | +0.0520 | +0.0071 |
| 6 | 450 | +0.0743 | +0.0557 |
| 7 | 450 | +0.0984 | +0.1325 |
| 8 | 450 | +0.1249 | +0.1365 |
| 9 | 450 | +0.1598 | +0.1899 |
| 10 | 450 | +0.2295 | +0.1597 |

### `persuadable` — Highest score — surcharge looks helpful on this row

- Offline τ̂ (extreme untouched-test row): `+0.4833`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_persuadable",
  "uplift_score": 0.48325663147098824,
  "recommended_treatment": "SURCHARGE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.48325663147098824,
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
  "distance_km": 14.697147643519855,
  "duration_sec": 3473.9905236887103,
  "hour_of_day": 20.0,
  "past_trips": 12.0,
  "avg_surge": 0.09574683800105177
}
```

### `neutral` — Score nearest zero

- Offline τ̂ (extreme untouched-test row): `-0.0000`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_neutral",
  "uplift_score": -6.941111831360036e-06,
  "recommended_treatment": "KEEP_QUOTE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -6.941111831360036e-06,
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
  "distance_km": 9.846403325752522,
  "duration_sec": 2077.828919348398,
  "hour_of_day": 6.0,
  "past_trips": 11.0,
  "avg_surge": 25.90308081129476
}
```

### `sleeping_dog` — Lowest score — surcharge looks harmful on this row

- Offline τ̂ (extreme untouched-test row): `-0.3478`. Planted effect on this same row: `+0.00`.
- HTTP `POST /predict_uplift` response:
```json
{
  "driver_id": "proof_sleeping_dog",
  "uplift_score": -0.347783721491809,
  "recommended_treatment": "NO_SURCHARGE",
  "ranking_supports_decision": false,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.347783721491809,
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
  "distance_km": 14.995991451467445,
  "duration_sec": 3102.861551960009,
  "hour_of_day": 10.0,
  "past_trips": 11.0,
  "avg_surge": 14.834943309589855
}
```

## What the lowest score does

Captured uplift `-0.3478` → `NO_SURCHARGE`. DPE keeps the surcharge. The calibration flag is false, so a score below -0.05 does not change the fare.
The planted effect on that same row is `+0.00`. The row was chosen because its score is the minimum, not because it belongs to the sleeping-dog cut.
The segment means above are on the untouched test. This row is an extreme score, not the segment mean.

## What the test intervals say

On these five seeds the untouched-test Qini interval for the T-learner stays above 0. The decision flag is false on every T-learner seed in this table. Where the calibration Qini lower bound is above 0, the false-override rate is still above the pre-registered maximum 0.10, so the rate check fails. Under the rule in PLAN.md this T-learner does not change the fare.
