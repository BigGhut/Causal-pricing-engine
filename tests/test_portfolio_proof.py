"""Light tests for portfolio proof helpers (no long-lived servers)."""

from __future__ import annotations

from scripts.portfolio_proof import dpe_policy_from_uplift


def test_dpe_policy_sleeping_dog_override():
    pol = dpe_policy_from_uplift(-0.12, threshold=0.05)
    assert pol["causal_override"] is True
    assert pol["causal_recommended_treatment"] == "NO_DISCOUNT_AVOID"
    assert "base fare" in pol["pricing_action"].lower() or "zero" in pol["pricing_action"].lower()


def test_dpe_policy_persuadable_no_override():
    pol = dpe_policy_from_uplift(0.2, threshold=0.05)
    assert pol["causal_override"] is False
    assert pol["causal_recommended_treatment"] == "DISCOUNT_10_PCT"
