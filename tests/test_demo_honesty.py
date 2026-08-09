"""Tests for honest demo scenario selection."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.demo import (
    DEFAULT_THRESHOLD,
    FEATURE_COLS,
    assert_scenario_honesty,
    pick_honest_scenarios,
    run_demo,
)


def test_pick_honest_scenarios_labels_match_scores():
    n = 20
    p = len(FEATURE_COLS)
    X = np.zeros((n, p), dtype=float)
    # Distinct rows so feature dicts differ
    X[:, 0] = np.arange(n)
    uplift = np.linspace(-0.3, 0.4, n)

    picks = pick_honest_scenarios(X, uplift, FEATURE_COLS, threshold=0.05)
    assert_scenario_honesty(picks, threshold=0.05)

    by_role = {p.role: p for p in picks}
    assert by_role["persuadable"].uplift == pytest.approx(0.4)
    assert by_role["sleeping_dog"].uplift == pytest.approx(-0.3)
    assert abs(by_role["neutral"].uplift) <= abs(by_role["persuadable"].uplift)
    assert abs(by_role["neutral"].uplift) <= abs(by_role["sleeping_dog"].uplift)

    # Treatment mapping agreement
    from src.api.main import recommend_treatment

    t_pos, _ = recommend_treatment(by_role["persuadable"].uplift, 0.05)
    t_neg, _ = recommend_treatment(by_role["sleeping_dog"].uplift, 0.05)
    assert t_pos == "DISCOUNT_10_PCT"
    assert t_neg == "NO_DISCOUNT_AVOID"


def test_pick_honest_scenarios_fails_without_negative_hte():
    X = np.zeros((10, len(FEATURE_COLS)))
    uplift = np.linspace(0.0, 0.5, 10)  # all non-negative
    with pytest.raises(RuntimeError, match="sleeping-dog"):
        pick_honest_scenarios(X, uplift, FEATURE_COLS, threshold=DEFAULT_THRESHOLD)


def test_run_demo_synthetic_exits_zero():
    assert run_demo(source="synthetic", threshold=DEFAULT_THRESHOLD) == 0
