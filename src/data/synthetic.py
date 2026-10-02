"""Synthetic offers where an additive surcharge is randomized at the offer.

The treatment is one thing everywhere in this repository:

- T = 1: the quote is the base fare plus an additive surcharge
- T = 0: the quote is the base fare
- Y: the driver accepts the offer

DPE's switchback (additive formula versus multiplicative formula, assigned by
virtual hour) is a different contrast and is not generated here.

The planted segments are thresholds on ``past_trips`` and ``distance_km``.
Both columns are in the model matrix, so a flexible model can memorize the
cuts. Recovering their sign is a pipeline check, not a discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Covariates known before the surcharge is applied. ``price`` and ``surge_bonus``
# are consequences of the quote, so they are not features.
FEATURE_COLUMNS: list[str] = [
    "distance_km",
    "duration_sec",
    "hour_of_day",
    "past_trips",
    "avg_surge",
]

TREATMENT_NAME = "additive_surcharge"
PERSUADABLE_TAU = 0.25
NEUTRAL_TAU = 0.0
SLEEPING_DOG_TAU = -0.12
PERSUADABLE_MIN_TRIPS = 8.0
PERSUADABLE_MAX_DISTANCE_KM = 8.0
SLEEPING_MAX_TRIPS = 5.0
SLEEPING_MIN_DISTANCE_KM = 9.0

SEGMENT_NOTE = (
    "Planted segments are thresholds on past_trips and distance_km, "
    "and both columns are given to the model. "
    f"persuadable: past_trips >= {PERSUADABLE_MIN_TRIPS:.0f} and "
    f"distance_km <= {PERSUADABLE_MAX_DISTANCE_KM:.0f}, tau = {PERSUADABLE_TAU:+.2f}. "
    f"sleeping_dog: past_trips <= {SLEEPING_MAX_TRIPS:.0f} and "
    f"distance_km >= {SLEEPING_MIN_DISTANCE_KM:.0f}, tau = {SLEEPING_DOG_TAU:+.2f}. "
    "Everyone else has tau = 0. "
    "A high score on these columns is not evidence that the model found a hidden group."
)


def assign_segment(past_trips: np.ndarray, distance_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return segment codes (2 persuadable, 1 neutral, 0 sleeping dog) and planted tau."""
    segment = np.ones(len(past_trips), dtype=float)
    persuadable = (past_trips >= PERSUADABLE_MIN_TRIPS) & (
        distance_km <= PERSUADABLE_MAX_DISTANCE_KM
    )
    sleeping = (past_trips <= SLEEPING_MAX_TRIPS) & (distance_km >= SLEEPING_MIN_DISTANCE_KM)
    segment[persuadable] = 2.0
    segment[sleeping] = 0.0
    planted = np.full(len(past_trips), NEUTRAL_TAU, dtype=float)
    planted[persuadable] = PERSUADABLE_TAU
    planted[sleeping] = SLEEPING_DOG_TAU
    return segment, planted


def generate_uplift_dataset(
    n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomize the additive surcharge independently of the covariates.

    ``hour_of_day`` is a covariate. It does not assign treatment.
    ``revenue`` is the fare the driver accepts: base fare, plus the surcharge
    only when the offer is treated and accepted. There is no discount.
    """
    rng = np.random.default_rng(random_state)

    distance_km = rng.uniform(1.0, 15.0, size=n)
    duration_sec = distance_km * rng.uniform(120.0, 240.0, size=n)
    hour_of_day = rng.integers(0, 24, size=n).astype(float)
    past_trips = rng.poisson(lam=8.0, size=n).astype(float)
    avg_surge = rng.uniform(0.0, 30.0, size=n)

    segment, true_uplift = assign_segment(past_trips, distance_km)
    treatment = rng.binomial(1, 0.5, size=n)

    baseline_logit = (
        -0.4
        + 0.03 * past_trips
        - 0.04 * distance_km
        - 0.005 * (hour_of_day - 12.0) ** 2
    )
    baseline_prob = 1.0 / (1.0 + np.exp(-baseline_logit))
    treated_prob = np.clip(baseline_prob + true_uplift, 0.01, 0.99)
    accept_prob = np.where(treatment == 1, treated_prob, baseline_prob)
    accepted = rng.binomial(1, accept_prob)

    base_fare = 120.0 + distance_km * 20.0 + duration_sec * 0.05
    surcharge_rub = 15.0 + 0.02 * duration_sec
    revenue = np.where(accepted == 1, base_fare + treatment * surcharge_rub, 0.0)

    return pd.DataFrame(
        {
            "driver_id": [f"d_{i:06d}" for i in range(n)],
            "distance_km": distance_km,
            "duration_sec": duration_sec,
            "hour_of_day": hour_of_day,
            "past_trips": past_trips,
            "avg_surge": avg_surge,
            "segment": segment,
            "true_uplift": true_uplift.astype(float),
            "treatment": treatment.astype(int),
            "accepted": accepted.astype(int),
            "base_fare": base_fare.astype(float),
            "surcharge_rub": surcharge_rub.astype(float),
            "revenue": revenue.astype(float),
        }
    )


def summarize_calibration(
    segment: np.ndarray,
    planted: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Mean predicted effect against the planted effect, by segment."""
    segment = np.asarray(segment, dtype=float).ravel()
    planted = np.asarray(planted, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    rows: list[dict[str, float | int | str]] = []
    for code, name in ((2.0, "persuadable"), (1.0, "neutral"), (0.0, "sleeping_dog")):
        mask = segment == code
        if not np.any(mask):
            continue
        pred_mean = float(np.mean(predicted[mask]))
        planted_mean = float(np.mean(planted[mask]))
        ratio = None if abs(planted_mean) < 1e-9 else float(pred_mean / planted_mean)
        rows.append(
            {
                "segment": name,
                "n": int(mask.sum()),
                "planted": planted_mean,
                "mean_predicted": pred_mean,
                "ratio": ratio if ratio is not None else float("nan"),
            }
        )
    return rows


def format_calibration(rows: list[dict]) -> str:
    """Plain-language calibration. A ratio far from 1 is an over- or under-shoot."""
    lines = [SEGMENT_NOTE]
    for row in rows:
        planted = float(row["planted"])
        predicted = float(row["mean_predicted"])
        if abs(planted) < 1e-9:
            ratio_text = "no ratio, the planted effect is 0"
        else:
            ratio_text = f"predicted/planted = {predicted / planted:.2f}"
        lines.append(
            f"  {row['segment']}: n={row['n']} planted {planted:+.2f} "
            f"mean τ̂ {predicted:+.3f} ({ratio_text})"
        )
    return "\n".join(lines)
