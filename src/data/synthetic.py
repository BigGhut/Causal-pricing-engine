"""Synthetic uplift datasets with heterogeneous treatment effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shared feature contract used by train script, API, and tests.
FEATURE_COLUMNS: list[str] = [
    "past_trips",
    "avg_surge",
    "price_sensitivity",
    "hour_of_day",
    "segment",
]


def generate_uplift_dataset(
    n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate observational uplift data with heterogeneous treatment effects.

    Segments (by ``segment`` feature, also driven by ``price_sensitivity``):
    - High sensitivity (segment 2): strong positive TE (persuadables).
    - Medium sensitivity (segment 1): near-zero TE (sure things / lost causes).
    - Low sensitivity (segment 0): negative TE (sleeping dogs).

    Args:
        n: Number of rows to generate.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with columns:
        ``user_id``, feature columns, ``treatment``, ``conversion``, ``revenue``.
    """
    rng = np.random.default_rng(random_state)

    past_trips = rng.poisson(lam=8.0, size=n).astype(float)
    avg_surge = rng.uniform(1.0, 2.5, size=n)
    price_sensitivity = rng.uniform(0.0, 1.0, size=n)
    hour_of_day = rng.integers(0, 24, size=n).astype(float)

    # Discrete segment for explicit heterogeneous TE
    segment = np.zeros(n, dtype=float)
    segment[price_sensitivity >= 0.66] = 2.0  # persuadables
    segment[(price_sensitivity >= 0.33) & (price_sensitivity < 0.66)] = 1.0  # neutral
    # segment < 0.33 stays 0 — sleeping dogs

    # Randomized treatment assignment (simulates A/B)
    treatment = rng.binomial(1, 0.5, size=n)

    # True ITE by segment
    true_uplift = np.where(
        segment == 2.0,
        0.25,
        np.where(segment == 1.0, 0.0, -0.12),
    )

    # Baseline conversion probability (control outcome)
    baseline_logit = (
        -0.8
        + 0.05 * past_trips
        - 0.3 * (avg_surge - 1.0)
        + 0.02 * (hour_of_day - 12.0)
    )
    baseline_prob = 1.0 / (1.0 + np.exp(-baseline_logit))
    treated_prob = np.clip(baseline_prob + true_uplift, 0.01, 0.99)

    conversion_prob = np.where(treatment == 1, treated_prob, baseline_prob)
    conversion = rng.binomial(1, conversion_prob)

    # Revenue: base fare-like amount, boosted on conversion and treatment
    base_revenue = 200.0 + 15.0 * past_trips + 50.0 * avg_surge
    revenue = np.where(
        conversion == 1,
        base_revenue * (1.0 - 0.05 * treatment),  # small discount cost when treated
        0.0,
    )
    revenue = revenue + rng.normal(0.0, 10.0, size=n)
    revenue = np.maximum(revenue, 0.0)

    df = pd.DataFrame(
        {
            "user_id": [f"u_{i:06d}" for i in range(n)],
            "past_trips": past_trips,
            "avg_surge": avg_surge,
            "price_sensitivity": price_sensitivity,
            "hour_of_day": hour_of_day,
            "segment": segment,
            "treatment": treatment.astype(int),
            "conversion": conversion.astype(int),
            "revenue": revenue.astype(float),
        }
    )
    return df
