"""Synthetic uplift datasets with heterogeneous treatment effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.dpe_connector import DPE_FEATURE_COLUMNS

# Unified canonical feature contract (aligned with DPE simulation schema)
FEATURE_COLUMNS: list[str] = list(DPE_FEATURE_COLUMNS)


def generate_uplift_dataset(
    n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic uplift data with heterogeneous treatment effects matching DPE feature schema.

    Features generated:
    - distance_km, duration_sec, price, surge_bonus, hour_of_day, past_trips, avg_surge

    Segments (for heterogeneity testing):
    - Persuadables (segment 2.0): high past_trips, high surge_bonus -> strong positive uplift (+0.25)
    - Neutral (segment 1.0): moderate past_trips -> near-zero uplift (~0.0)
    - Sleeping dogs (segment 0.0): low past_trips, high surge_bonus -> negative uplift (-0.12)

    Args:
        n: Number of rows to generate.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with columns:
        ``user_id``, DPE_FEATURE_COLUMNS, ``segment``, ``treatment``, ``conversion``, ``revenue``.
    """
    rng = np.random.default_rng(random_state)

    distance_km = rng.uniform(1.0, 15.0, size=n)
    duration_sec = distance_km * rng.uniform(120.0, 240.0, size=n)
    price = 150.0 + distance_km * 25.0 + duration_sec * 0.1
    surge_bonus = rng.uniform(0.0, 50.0, size=n)
    hour_of_day = rng.integers(0, 24, size=n).astype(float)
    past_trips = rng.poisson(lam=8.0, size=n).astype(float)
    avg_surge = rng.uniform(0.0, 30.0, size=n)

    # Segment definition for heterogeneity tests
    segment = np.ones(n, dtype=float)  # 1.0 = neutral default
    persuadable_mask = (past_trips >= 8.0) & (surge_bonus >= 15.0)
    sleeping_dog_mask = (past_trips < 4.0) & (surge_bonus >= 20.0)
    segment[persuadable_mask] = 2.0
    segment[sleeping_dog_mask] = 0.0

    # Treatment assignment (50/50 randomized A/B)
    treatment = rng.binomial(1, 0.5, size=n)

    # True ITE by segment
    true_uplift = np.where(
        segment == 2.0,
        0.25,
        np.where(segment == 1.0, 0.0, -0.12),
    )

    # Baseline conversion probability (control outcome)
    baseline_logit = (
        -0.5
        + 0.04 * past_trips
        - 0.05 * distance_km
        + 0.01 * surge_bonus
        - 0.01 * (hour_of_day - 12.0) ** 2
    )
    baseline_prob = 1.0 / (1.0 + np.exp(-baseline_logit))
    treated_prob = np.clip(baseline_prob + true_uplift, 0.01, 0.99)

    conversion_prob = np.where(treatment == 1, treated_prob, baseline_prob)
    conversion = rng.binomial(1, conversion_prob)

    base_revenue = price + surge_bonus
    revenue = np.where(
        conversion == 1,
        base_revenue * (1.0 - 0.05 * treatment),
        0.0,
    )
    revenue = np.maximum(revenue + rng.normal(0.0, 5.0, size=n), 0.0)

    df = pd.DataFrame(
        {
            "user_id": [f"u_{i:06d}" for i in range(n)],
            "distance_km": distance_km,
            "duration_sec": duration_sec,
            "price": price,
            "surge_bonus": surge_bonus,
            "hour_of_day": hour_of_day,
            "past_trips": past_trips,
            "avg_surge": avg_surge,
            "segment": segment,
            "treatment": treatment.astype(int),
            "conversion": conversion.astype(int),
            "revenue": revenue.astype(float),
        }
    )
    return df
