# Portfolio proof evidence

_Generated automatically by `scripts/portfolio_proof.py` at **2026-10-02 15:30:39 UTC**._

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
- CPE `/health`: `{"status": "ok", "service": "causal-pricing-engine", "model_loaded": "true", "model_name": "t_learner", "source": "portfolio_proof_seed_42_n_30000", "feature_columns": ["distance_km", "duration_sec", "hour_of_day", "past_trips", "avg_surge"], "ranking_supports_decision": true}`

## All five seeds

| Seed | Learner | θ* | Cal false | Cal caught | Test false | Test caught | Flag | Test Qini | Test 95% | PEHE |
|:---|:---|---:|---:|---:|---:|---:|:---|---:|:---|---:|
| 42 | t_learner | 0.10 | 0.0968 | 0.5437 | 0.0996 | 0.5293 | true | +0.0985 | [+0.0716, +0.1253] | 0.0512 |
| 42 | x_learner | 0.09 | 0.0439 | 0.5291 | 0.0657 | 0.5064 | true | +0.0838 | [+0.0566, +0.1109] | 0.0818 |
| 7 | t_learner | none | undefined | undefined | undefined | undefined | false | +0.0946 | [+0.0668, +0.1221] | 0.0531 |
| 7 | x_learner | 0.13 | 0.0709 | 0.3786 | 0.0342 | 0.3863 | true | +0.1037 | [+0.0779, +0.1301] | 0.0797 |
| 11 | t_learner | 0.08 | 0.0932 | 0.6801 | 0.0629 | 0.7109 | true | +0.0970 | [+0.0691, +0.1238] | 0.0487 |
| 11 | x_learner | 0.09 | 0.0087 | 0.3065 | 0.0583 | 0.2997 | true | +0.0927 | [+0.0637, +0.1181] | 0.0807 |
| 99 | t_learner | 0.36 | 0.0000 | 0.0026 | 0.5000 | 0.0026 | false | +0.0917 | [+0.0630, +0.1183] | 0.0501 |
| 99 | x_learner | 0.11 | 0.0909 | 0.4627 | 0.0979 | 0.5436 | true | +0.0890 | [+0.0618, +0.1170] | 0.0796 |
| 123 | t_learner | none | undefined | undefined | undefined | undefined | false | +0.0915 | [+0.0634, +0.1172] | 0.0489 |
| 123 | x_learner | 0.09 | 0.0355 | 0.4644 | 0.0539 | 0.4647 | true | +0.0890 | [+0.0615, +0.1147] | 0.0788 |

- T-learner test Qini spread: `+0.0915 … +0.0985`.
- T-learner test PEHE spread: `+0.0487 … +0.0531`.
- X-learner test Qini spread: `+0.0838 … +0.1037`.
- X-learner test PEHE spread: `+0.0788 … +0.0818`.

## Pre-registered seed, untouched test

Generation seed `42`, T-learner. Calibration n=`4500`, test n=`4500`.
- Calibration Qini `+0.0993`, 95% `[+0.0705, +0.1257]` on 1000 resamples. Chosen θ* `0.10`. Calibration false rate `0.0968`, caught rate `0.5437`. Test false rate `0.0996`, test caught rate `0.5293`. Flag `true`.
- Untouched-test Qini `+0.0985`, 95% `[+0.0716, +0.1253]` on 1000 resamples. This slice did not set the flag.
- Random-score Qini on that test: mean `+0.0002`, 95% `[-0.0267, +0.0279]` over 1000 draws.
- PEHE `0.0512`.
- Test false-override rate `0.0996`.
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

## Calibration curve: false rate against caught rate

Each row is one θ on the grid 0.00, 0.01, …, 0.50.
Override means score < −θ. Caught rate is the share of rows with true_uplift < 0 that the override hits.
The chosen mark is the calibration rule. The untouched test is not used to place that mark.

### Seed `42`, `t_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1458 | 0.7229 | 0.9806 |  |
| 0.01 | 1179 | 0.6607 | 0.9709 |  |
| 0.02 | 944 | 0.5805 | 0.9612 |  |
| 0.03 | 758 | 0.4881 | 0.9417 |  |
| 0.04 | 644 | 0.4115 | 0.9199 |  |
| 0.05 | 549 | 0.3260 | 0.8981 |  |
| 0.06 | 460 | 0.2370 | 0.8519 |  |
| 0.07 | 414 | 0.1787 | 0.8252 |  |
| 0.08 | 362 | 0.1381 | 0.7573 |  |
| 0.09 | 307 | 0.1042 | 0.6675 |  |
| 0.10 | 248 | 0.0968 | 0.5437 | yes |
| 0.11 | 189 | 0.1005 | 0.4126 |  |
| 0.12 | 152 | 0.1118 | 0.3277 |  |
| 0.13 | 128 | 0.1250 | 0.2718 |  |
| 0.14 | 89 | 0.1798 | 0.1772 |  |
| 0.15 | 57 | 0.2632 | 0.1019 |  |
| 0.16 | 39 | 0.3590 | 0.0607 |  |
| 0.17 | 24 | 0.5833 | 0.0243 |  |
| 0.18 | 17 | 0.7059 | 0.0121 |  |
| 0.19 | 16 | 0.6875 | 0.0121 |  |
| 0.20 | 13 | 0.6923 | 0.0097 |  |
| 0.21 | 11 | 0.7273 | 0.0073 |  |
| 0.22 | 9 | 0.7778 | 0.0049 |  |
| 0.23 | 7 | 0.7143 | 0.0049 |  |
| 0.24 | 7 | 0.7143 | 0.0049 |  |
| 0.25 | 7 | 0.7143 | 0.0049 |  |
| 0.26 | 7 | 0.7143 | 0.0049 |  |
| 0.27 | 7 | 0.7143 | 0.0049 |  |
| 0.28 | 7 | 0.7143 | 0.0049 |  |
| 0.29 | 7 | 0.7143 | 0.0049 |  |
| 0.30 | 7 | 0.7143 | 0.0049 |  |
| 0.31 | 6 | 0.8333 | 0.0024 |  |
| 0.32 | 4 | 1.0000 | 0.0000 |  |
| 0.33 | 4 | 1.0000 | 0.0000 |  |
| 0.34 | 4 | 1.0000 | 0.0000 |  |
| 0.35 | 4 | 1.0000 | 0.0000 |  |
| 0.36 | 4 | 1.0000 | 0.0000 |  |
| 0.37 | 4 | 1.0000 | 0.0000 |  |
| 0.38 | 3 | 1.0000 | 0.0000 |  |
| 0.39 | 2 | 1.0000 | 0.0000 |  |
| 0.40 | 2 | 1.0000 | 0.0000 |  |
| 0.41 | 2 | 1.0000 | 0.0000 |  |
| 0.42 | 1 | 1.0000 | 0.0000 |  |
| 0.43 | 1 | 1.0000 | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `42`, `x_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1124 | 0.6335 | 1.0000 |  |
| 0.01 | 979 | 0.5792 | 1.0000 |  |
| 0.02 | 852 | 0.5223 | 0.9879 |  |
| 0.03 | 737 | 0.4586 | 0.9684 |  |
| 0.04 | 615 | 0.3870 | 0.9150 |  |
| 0.05 | 511 | 0.3190 | 0.8447 |  |
| 0.06 | 437 | 0.2700 | 0.7743 |  |
| 0.07 | 361 | 0.1717 | 0.7257 |  |
| 0.08 | 290 | 0.1138 | 0.6238 |  |
| 0.09 | 228 | 0.0439 | 0.5291 | yes |
| 0.10 | 180 | 0.0167 | 0.4296 |  |
| 0.11 | 130 | 0.0077 | 0.3131 |  |
| 0.12 | 91 | 0.0110 | 0.2184 |  |
| 0.13 | 64 | 0.0000 | 0.1553 |  |
| 0.14 | 42 | 0.0000 | 0.1019 |  |
| 0.15 | 23 | 0.0000 | 0.0558 |  |
| 0.16 | 15 | 0.0000 | 0.0364 |  |
| 0.17 | 10 | 0.0000 | 0.0243 |  |
| 0.18 | 7 | 0.0000 | 0.0170 |  |
| 0.19 | 2 | 0.0000 | 0.0049 |  |
| 0.20 | 1 | 0.0000 | 0.0024 |  |
| 0.21 | 0 | undefined | 0.0000 |  |
| 0.22 | 0 | undefined | 0.0000 |  |
| 0.23 | 0 | undefined | 0.0000 |  |
| 0.24 | 0 | undefined | 0.0000 |  |
| 0.25 | 0 | undefined | 0.0000 |  |
| 0.26 | 0 | undefined | 0.0000 |  |
| 0.27 | 0 | undefined | 0.0000 |  |
| 0.28 | 0 | undefined | 0.0000 |  |
| 0.29 | 0 | undefined | 0.0000 |  |
| 0.30 | 0 | undefined | 0.0000 |  |
| 0.31 | 0 | undefined | 0.0000 |  |
| 0.32 | 0 | undefined | 0.0000 |  |
| 0.33 | 0 | undefined | 0.0000 |  |
| 0.34 | 0 | undefined | 0.0000 |  |
| 0.35 | 0 | undefined | 0.0000 |  |
| 0.36 | 0 | undefined | 0.0000 |  |
| 0.37 | 0 | undefined | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `7`, `t_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 2117 | 0.8375 | 0.9942 |  |
| 0.01 | 1828 | 0.8118 | 0.9942 |  |
| 0.02 | 1484 | 0.7689 | 0.9913 |  |
| 0.03 | 1185 | 0.7105 | 0.9913 |  |
| 0.04 | 975 | 0.6503 | 0.9855 |  |
| 0.05 | 821 | 0.5932 | 0.9653 |  |
| 0.06 | 675 | 0.5289 | 0.9191 |  |
| 0.07 | 533 | 0.4296 | 0.8786 |  |
| 0.08 | 447 | 0.3624 | 0.8237 |  |
| 0.09 | 385 | 0.3039 | 0.7746 |  |
| 0.10 | 334 | 0.2545 | 0.7197 |  |
| 0.11 | 281 | 0.2313 | 0.6243 |  |
| 0.12 | 209 | 0.2536 | 0.4509 |  |
| 0.13 | 170 | 0.2353 | 0.3757 |  |
| 0.14 | 125 | 0.2880 | 0.2572 |  |
| 0.15 | 81 | 0.3580 | 0.1503 |  |
| 0.16 | 47 | 0.3830 | 0.0838 |  |
| 0.17 | 22 | 0.5455 | 0.0289 |  |
| 0.18 | 17 | 0.5882 | 0.0202 |  |
| 0.19 | 15 | 0.6000 | 0.0173 |  |
| 0.20 | 15 | 0.6000 | 0.0173 |  |
| 0.21 | 13 | 0.6154 | 0.0145 |  |
| 0.22 | 9 | 0.5556 | 0.0116 |  |
| 0.23 | 8 | 0.6250 | 0.0087 |  |
| 0.24 | 7 | 0.5714 | 0.0087 |  |
| 0.25 | 4 | 0.7500 | 0.0029 |  |
| 0.26 | 4 | 0.7500 | 0.0029 |  |
| 0.27 | 4 | 0.7500 | 0.0029 |  |
| 0.28 | 3 | 0.6667 | 0.0029 |  |
| 0.29 | 3 | 0.6667 | 0.0029 |  |
| 0.30 | 2 | 0.5000 | 0.0029 |  |
| 0.31 | 2 | 0.5000 | 0.0029 |  |
| 0.32 | 1 | 1.0000 | 0.0000 |  |
| 0.33 | 1 | 1.0000 | 0.0000 |  |
| 0.34 | 1 | 1.0000 | 0.0000 |  |
| 0.35 | 1 | 1.0000 | 0.0000 |  |
| 0.36 | 1 | 1.0000 | 0.0000 |  |
| 0.37 | 1 | 1.0000 | 0.0000 |  |
| 0.38 | 1 | 1.0000 | 0.0000 |  |
| 0.39 | 1 | 1.0000 | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `7`, `x_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1453 | 0.7619 | 1.0000 |  |
| 0.01 | 1303 | 0.7345 | 1.0000 |  |
| 0.02 | 1141 | 0.6968 | 1.0000 |  |
| 0.03 | 992 | 0.6512 | 1.0000 |  |
| 0.04 | 861 | 0.6051 | 0.9827 |  |
| 0.05 | 726 | 0.5551 | 0.9335 |  |
| 0.06 | 625 | 0.5088 | 0.8873 |  |
| 0.07 | 526 | 0.4639 | 0.8150 |  |
| 0.08 | 437 | 0.3982 | 0.7601 |  |
| 0.09 | 350 | 0.3229 | 0.6850 |  |
| 0.10 | 282 | 0.2305 | 0.6272 |  |
| 0.11 | 229 | 0.1921 | 0.5347 |  |
| 0.12 | 176 | 0.1193 | 0.4480 |  |
| 0.13 | 141 | 0.0709 | 0.3786 | yes |
| 0.14 | 94 | 0.0000 | 0.2717 |  |
| 0.15 | 66 | 0.0000 | 0.1908 |  |
| 0.16 | 42 | 0.0000 | 0.1214 |  |
| 0.17 | 28 | 0.0000 | 0.0809 |  |
| 0.18 | 21 | 0.0000 | 0.0607 |  |
| 0.19 | 12 | 0.0000 | 0.0347 |  |
| 0.20 | 9 | 0.0000 | 0.0260 |  |
| 0.21 | 6 | 0.0000 | 0.0173 |  |
| 0.22 | 2 | 0.0000 | 0.0058 |  |
| 0.23 | 1 | 0.0000 | 0.0029 |  |
| 0.24 | 1 | 0.0000 | 0.0029 |  |
| 0.25 | 1 | 0.0000 | 0.0029 |  |
| 0.26 | 1 | 0.0000 | 0.0029 |  |
| 0.27 | 0 | undefined | 0.0000 |  |
| 0.28 | 0 | undefined | 0.0000 |  |
| 0.29 | 0 | undefined | 0.0000 |  |
| 0.30 | 0 | undefined | 0.0000 |  |
| 0.31 | 0 | undefined | 0.0000 |  |
| 0.32 | 0 | undefined | 0.0000 |  |
| 0.33 | 0 | undefined | 0.0000 |  |
| 0.34 | 0 | undefined | 0.0000 |  |
| 0.35 | 0 | undefined | 0.0000 |  |
| 0.36 | 0 | undefined | 0.0000 |  |
| 0.37 | 0 | undefined | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `11`, `t_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1147 | 0.6861 | 0.9677 |  |
| 0.01 | 896 | 0.6004 | 0.9624 |  |
| 0.02 | 724 | 0.5097 | 0.9543 |  |
| 0.03 | 635 | 0.4472 | 0.9435 |  |
| 0.04 | 558 | 0.3853 | 0.9220 |  |
| 0.05 | 468 | 0.2842 | 0.9005 |  |
| 0.06 | 384 | 0.1953 | 0.8306 |  |
| 0.07 | 325 | 0.1323 | 0.7581 |  |
| 0.08 | 279 | 0.0932 | 0.6801 | yes |
| 0.09 | 224 | 0.0670 | 0.5618 |  |
| 0.10 | 133 | 0.0902 | 0.3253 |  |
| 0.11 | 59 | 0.1695 | 0.1317 |  |
| 0.12 | 34 | 0.1765 | 0.0753 |  |
| 0.13 | 18 | 0.2778 | 0.0349 |  |
| 0.14 | 12 | 0.4167 | 0.0188 |  |
| 0.15 | 7 | 0.4286 | 0.0108 |  |
| 0.16 | 4 | 0.5000 | 0.0054 |  |
| 0.17 | 3 | 0.3333 | 0.0054 |  |
| 0.18 | 3 | 0.3333 | 0.0054 |  |
| 0.19 | 2 | 0.5000 | 0.0027 |  |
| 0.20 | 2 | 0.5000 | 0.0027 |  |
| 0.21 | 2 | 0.5000 | 0.0027 |  |
| 0.22 | 2 | 0.5000 | 0.0027 |  |
| 0.23 | 2 | 0.5000 | 0.0027 |  |
| 0.24 | 2 | 0.5000 | 0.0027 |  |
| 0.25 | 2 | 0.5000 | 0.0027 |  |
| 0.26 | 2 | 0.5000 | 0.0027 |  |
| 0.27 | 2 | 0.5000 | 0.0027 |  |
| 0.28 | 2 | 0.5000 | 0.0027 |  |
| 0.29 | 2 | 0.5000 | 0.0027 |  |
| 0.30 | 1 | 0.0000 | 0.0027 |  |
| 0.31 | 1 | 0.0000 | 0.0027 |  |
| 0.32 | 1 | 0.0000 | 0.0027 |  |
| 0.33 | 1 | 0.0000 | 0.0027 |  |
| 0.34 | 1 | 0.0000 | 0.0027 |  |
| 0.35 | 1 | 0.0000 | 0.0027 |  |
| 0.36 | 1 | 0.0000 | 0.0027 |  |
| 0.37 | 1 | 0.0000 | 0.0027 |  |
| 0.38 | 1 | 0.0000 | 0.0027 |  |
| 0.39 | 1 | 0.0000 | 0.0027 |  |
| 0.40 | 1 | 0.0000 | 0.0027 |  |
| 0.41 | 1 | 0.0000 | 0.0027 |  |
| 0.42 | 1 | 0.0000 | 0.0027 |  |
| 0.43 | 1 | 0.0000 | 0.0027 |  |
| 0.44 | 1 | 0.0000 | 0.0027 |  |
| 0.45 | 1 | 0.0000 | 0.0027 |  |
| 0.46 | 1 | 0.0000 | 0.0027 |  |
| 0.47 | 1 | 0.0000 | 0.0027 |  |
| 0.48 | 1 | 0.0000 | 0.0027 |  |
| 0.49 | 1 | 0.0000 | 0.0027 |  |
| 0.50 | 1 | 0.0000 | 0.0027 |  |

### Seed `11`, `x_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1052 | 0.6464 | 1.0000 |  |
| 0.01 | 906 | 0.6071 | 0.9570 |  |
| 0.02 | 776 | 0.5541 | 0.9301 |  |
| 0.03 | 638 | 0.4984 | 0.8602 |  |
| 0.04 | 531 | 0.4463 | 0.7903 |  |
| 0.05 | 407 | 0.3661 | 0.6935 |  |
| 0.06 | 319 | 0.2978 | 0.6022 |  |
| 0.07 | 231 | 0.1991 | 0.4973 |  |
| 0.08 | 160 | 0.1375 | 0.3710 |  |
| 0.09 | 115 | 0.0087 | 0.3065 | yes |
| 0.10 | 75 | 0.0000 | 0.2016 |  |
| 0.11 | 45 | 0.0000 | 0.1210 |  |
| 0.12 | 27 | 0.0000 | 0.0726 |  |
| 0.13 | 15 | 0.0000 | 0.0403 |  |
| 0.14 | 7 | 0.0000 | 0.0188 |  |
| 0.15 | 4 | 0.0000 | 0.0108 |  |
| 0.16 | 3 | 0.0000 | 0.0081 |  |
| 0.17 | 2 | 0.0000 | 0.0054 |  |
| 0.18 | 1 | 0.0000 | 0.0027 |  |
| 0.19 | 1 | 0.0000 | 0.0027 |  |
| 0.20 | 0 | undefined | 0.0000 |  |
| 0.21 | 0 | undefined | 0.0000 |  |
| 0.22 | 0 | undefined | 0.0000 |  |
| 0.23 | 0 | undefined | 0.0000 |  |
| 0.24 | 0 | undefined | 0.0000 |  |
| 0.25 | 0 | undefined | 0.0000 |  |
| 0.26 | 0 | undefined | 0.0000 |  |
| 0.27 | 0 | undefined | 0.0000 |  |
| 0.28 | 0 | undefined | 0.0000 |  |
| 0.29 | 0 | undefined | 0.0000 |  |
| 0.30 | 0 | undefined | 0.0000 |  |
| 0.31 | 0 | undefined | 0.0000 |  |
| 0.32 | 0 | undefined | 0.0000 |  |
| 0.33 | 0 | undefined | 0.0000 |  |
| 0.34 | 0 | undefined | 0.0000 |  |
| 0.35 | 0 | undefined | 0.0000 |  |
| 0.36 | 0 | undefined | 0.0000 |  |
| 0.37 | 0 | undefined | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `99`, `t_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1950 | 0.8041 | 0.9820 |  |
| 0.01 | 1690 | 0.7740 | 0.9820 |  |
| 0.02 | 1405 | 0.7288 | 0.9794 |  |
| 0.03 | 1184 | 0.6807 | 0.9717 |  |
| 0.04 | 992 | 0.6260 | 0.9537 |  |
| 0.05 | 830 | 0.5614 | 0.9357 |  |
| 0.06 | 680 | 0.4868 | 0.8972 |  |
| 0.07 | 573 | 0.4188 | 0.8560 |  |
| 0.08 | 482 | 0.3714 | 0.7789 |  |
| 0.09 | 403 | 0.3375 | 0.6864 |  |
| 0.10 | 340 | 0.3088 | 0.6041 |  |
| 0.11 | 276 | 0.2826 | 0.5090 |  |
| 0.12 | 229 | 0.2620 | 0.4344 |  |
| 0.13 | 188 | 0.2606 | 0.3573 |  |
| 0.14 | 157 | 0.2611 | 0.2982 |  |
| 0.15 | 120 | 0.3000 | 0.2159 |  |
| 0.16 | 91 | 0.3187 | 0.1594 |  |
| 0.17 | 64 | 0.3438 | 0.1080 |  |
| 0.18 | 46 | 0.4348 | 0.0668 |  |
| 0.19 | 33 | 0.5455 | 0.0386 |  |
| 0.20 | 24 | 0.6250 | 0.0231 |  |
| 0.21 | 20 | 0.6500 | 0.0180 |  |
| 0.22 | 16 | 0.6875 | 0.0129 |  |
| 0.23 | 13 | 0.6923 | 0.0103 |  |
| 0.24 | 11 | 0.7273 | 0.0077 |  |
| 0.25 | 11 | 0.7273 | 0.0077 |  |
| 0.26 | 10 | 0.7000 | 0.0077 |  |
| 0.27 | 10 | 0.7000 | 0.0077 |  |
| 0.28 | 7 | 0.5714 | 0.0077 |  |
| 0.29 | 7 | 0.5714 | 0.0077 |  |
| 0.30 | 4 | 0.7500 | 0.0026 |  |
| 0.31 | 3 | 0.6667 | 0.0026 |  |
| 0.32 | 2 | 0.5000 | 0.0026 |  |
| 0.33 | 2 | 0.5000 | 0.0026 |  |
| 0.34 | 2 | 0.5000 | 0.0026 |  |
| 0.35 | 2 | 0.5000 | 0.0026 |  |
| 0.36 | 1 | 0.0000 | 0.0026 | yes |
| 0.37 | 1 | 0.0000 | 0.0026 |  |
| 0.38 | 1 | 0.0000 | 0.0026 |  |
| 0.39 | 1 | 0.0000 | 0.0026 |  |
| 0.40 | 1 | 0.0000 | 0.0026 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `99`, `x_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1441 | 0.7300 | 1.0000 |  |
| 0.01 | 1307 | 0.7024 | 1.0000 |  |
| 0.02 | 1139 | 0.6585 | 1.0000 |  |
| 0.03 | 998 | 0.6132 | 0.9923 |  |
| 0.04 | 854 | 0.5539 | 0.9794 |  |
| 0.05 | 734 | 0.5000 | 0.9434 |  |
| 0.06 | 623 | 0.4462 | 0.8869 |  |
| 0.07 | 524 | 0.3855 | 0.8278 |  |
| 0.08 | 433 | 0.3256 | 0.7506 |  |
| 0.09 | 332 | 0.2530 | 0.6375 |  |
| 0.10 | 260 | 0.1923 | 0.5398 |  |
| 0.11 | 198 | 0.0909 | 0.4627 | yes |
| 0.12 | 144 | 0.0486 | 0.3522 |  |
| 0.13 | 99 | 0.0303 | 0.2468 |  |
| 0.14 | 77 | 0.0000 | 0.1979 |  |
| 0.15 | 57 | 0.0000 | 0.1465 |  |
| 0.16 | 35 | 0.0000 | 0.0900 |  |
| 0.17 | 20 | 0.0000 | 0.0514 |  |
| 0.18 | 15 | 0.0000 | 0.0386 |  |
| 0.19 | 12 | 0.0000 | 0.0308 |  |
| 0.20 | 7 | 0.0000 | 0.0180 |  |
| 0.21 | 2 | 0.0000 | 0.0051 |  |
| 0.22 | 1 | 0.0000 | 0.0026 |  |
| 0.23 | 0 | undefined | 0.0000 |  |
| 0.24 | 0 | undefined | 0.0000 |  |
| 0.25 | 0 | undefined | 0.0000 |  |
| 0.26 | 0 | undefined | 0.0000 |  |
| 0.27 | 0 | undefined | 0.0000 |  |
| 0.28 | 0 | undefined | 0.0000 |  |
| 0.29 | 0 | undefined | 0.0000 |  |
| 0.30 | 0 | undefined | 0.0000 |  |
| 0.31 | 0 | undefined | 0.0000 |  |
| 0.32 | 0 | undefined | 0.0000 |  |
| 0.33 | 0 | undefined | 0.0000 |  |
| 0.34 | 0 | undefined | 0.0000 |  |
| 0.35 | 0 | undefined | 0.0000 |  |
| 0.36 | 0 | undefined | 0.0000 |  |
| 0.37 | 0 | undefined | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `123`, `t_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1709 | 0.7970 | 0.9886 |  |
| 0.01 | 1404 | 0.7528 | 0.9886 |  |
| 0.02 | 1138 | 0.6968 | 0.9829 |  |
| 0.03 | 906 | 0.6269 | 0.9630 |  |
| 0.04 | 757 | 0.5627 | 0.9430 |  |
| 0.05 | 633 | 0.4897 | 0.9202 |  |
| 0.06 | 532 | 0.4323 | 0.8604 |  |
| 0.07 | 460 | 0.3870 | 0.8034 |  |
| 0.08 | 373 | 0.3539 | 0.6866 |  |
| 0.09 | 286 | 0.3392 | 0.5385 |  |
| 0.10 | 211 | 0.3412 | 0.3960 |  |
| 0.11 | 144 | 0.3750 | 0.2564 |  |
| 0.12 | 96 | 0.4062 | 0.1624 |  |
| 0.13 | 67 | 0.4627 | 0.1026 |  |
| 0.14 | 51 | 0.4706 | 0.0769 |  |
| 0.15 | 35 | 0.5429 | 0.0456 |  |
| 0.16 | 22 | 0.5455 | 0.0285 |  |
| 0.17 | 16 | 0.5625 | 0.0199 |  |
| 0.18 | 15 | 0.6000 | 0.0171 |  |
| 0.19 | 8 | 0.6250 | 0.0085 |  |
| 0.20 | 6 | 0.8333 | 0.0028 |  |
| 0.21 | 6 | 0.8333 | 0.0028 |  |
| 0.22 | 6 | 0.8333 | 0.0028 |  |
| 0.23 | 5 | 0.8000 | 0.0028 |  |
| 0.24 | 4 | 0.7500 | 0.0028 |  |
| 0.25 | 3 | 0.6667 | 0.0028 |  |
| 0.26 | 3 | 0.6667 | 0.0028 |  |
| 0.27 | 2 | 1.0000 | 0.0000 |  |
| 0.28 | 2 | 1.0000 | 0.0000 |  |
| 0.29 | 2 | 1.0000 | 0.0000 |  |
| 0.30 | 1 | 1.0000 | 0.0000 |  |
| 0.31 | 1 | 1.0000 | 0.0000 |  |
| 0.32 | 1 | 1.0000 | 0.0000 |  |
| 0.33 | 1 | 1.0000 | 0.0000 |  |
| 0.34 | 1 | 1.0000 | 0.0000 |  |
| 0.35 | 1 | 1.0000 | 0.0000 |  |
| 0.36 | 1 | 1.0000 | 0.0000 |  |
| 0.37 | 1 | 1.0000 | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

### Seed `123`, `x_learner`

| θ | n fired | false rate | caught rate | chosen |
|---:|---:|---:|---:|:---|
| 0.00 | 1157 | 0.6966 | 1.0000 |  |
| 0.01 | 988 | 0.6447 | 1.0000 |  |
| 0.02 | 829 | 0.5778 | 0.9972 |  |
| 0.03 | 686 | 0.5087 | 0.9601 |  |
| 0.04 | 567 | 0.4550 | 0.8803 |  |
| 0.05 | 476 | 0.3845 | 0.8348 |  |
| 0.06 | 372 | 0.2769 | 0.7664 |  |
| 0.07 | 295 | 0.2000 | 0.6724 |  |
| 0.08 | 228 | 0.1228 | 0.5698 |  |
| 0.09 | 169 | 0.0355 | 0.4644 | yes |
| 0.10 | 123 | 0.0081 | 0.3476 |  |
| 0.11 | 82 | 0.0000 | 0.2336 |  |
| 0.12 | 58 | 0.0000 | 0.1652 |  |
| 0.13 | 38 | 0.0000 | 0.1083 |  |
| 0.14 | 27 | 0.0000 | 0.0769 |  |
| 0.15 | 20 | 0.0000 | 0.0570 |  |
| 0.16 | 9 | 0.0000 | 0.0256 |  |
| 0.17 | 6 | 0.0000 | 0.0171 |  |
| 0.18 | 4 | 0.0000 | 0.0114 |  |
| 0.19 | 0 | undefined | 0.0000 |  |
| 0.20 | 0 | undefined | 0.0000 |  |
| 0.21 | 0 | undefined | 0.0000 |  |
| 0.22 | 0 | undefined | 0.0000 |  |
| 0.23 | 0 | undefined | 0.0000 |  |
| 0.24 | 0 | undefined | 0.0000 |  |
| 0.25 | 0 | undefined | 0.0000 |  |
| 0.26 | 0 | undefined | 0.0000 |  |
| 0.27 | 0 | undefined | 0.0000 |  |
| 0.28 | 0 | undefined | 0.0000 |  |
| 0.29 | 0 | undefined | 0.0000 |  |
| 0.30 | 0 | undefined | 0.0000 |  |
| 0.31 | 0 | undefined | 0.0000 |  |
| 0.32 | 0 | undefined | 0.0000 |  |
| 0.33 | 0 | undefined | 0.0000 |  |
| 0.34 | 0 | undefined | 0.0000 |  |
| 0.35 | 0 | undefined | 0.0000 |  |
| 0.36 | 0 | undefined | 0.0000 |  |
| 0.37 | 0 | undefined | 0.0000 |  |
| 0.38 | 0 | undefined | 0.0000 |  |
| 0.39 | 0 | undefined | 0.0000 |  |
| 0.40 | 0 | undefined | 0.0000 |  |
| 0.41 | 0 | undefined | 0.0000 |  |
| 0.42 | 0 | undefined | 0.0000 |  |
| 0.43 | 0 | undefined | 0.0000 |  |
| 0.44 | 0 | undefined | 0.0000 |  |
| 0.45 | 0 | undefined | 0.0000 |  |
| 0.46 | 0 | undefined | 0.0000 |  |
| 0.47 | 0 | undefined | 0.0000 |  |
| 0.48 | 0 | undefined | 0.0000 |  |
| 0.49 | 0 | undefined | 0.0000 |  |
| 0.50 | 0 | undefined | 0.0000 |  |

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
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": 0.48325663147098824,
  "causal_override": false,
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
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
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -6.941111831360036e-06,
  "causal_override": false,
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
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
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
  "model_name": "t_learner"
}
```
- DPE policy simulation from that score:
```json
{
  "causal_uplift_score": -0.347783721491809,
  "causal_override": true,
  "ranking_supports_decision": true,
  "score_threshold": 0.1,
  "causal_recommended_treatment": "NO_SURCHARGE",
  "test_group_if_dpe": "CAUSAL_NO_SURGE",
  "pricing_action": "charge the base fare and drop the additive surcharge"
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

Captured uplift `-0.3478` → `NO_SURCHARGE`. DPE would charge the base fare because the version-2 flag is true at the frozen threshold. A row with true effect at least 0 is one of the false overrides still inside the 0.10 cap.
The planted effect on that same row is `+0.00`. The row was chosen because its score is the minimum, not because it belongs to the sleeping-dog cut.
The segment means above are on the untouched test. This row is an extreme score, not the segment mean.

## What the test intervals say

On these five seeds the untouched-test Qini interval for the T-learner stays above 0. The flag is true on 2 of 5 T-learner seeds: seed 42 θ*=0.10 test false=0.0996 test caught=0.5293, seed 11 θ*=0.08 test false=0.0629 test caught=0.7109. The other seeds stay in the table and are not replaced by these. A calibration point that barely catches sleeping dogs is not useful: seed 99 caught 0.0026.
