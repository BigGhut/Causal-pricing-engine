"""Light tests for portfolio proof helpers (no long-lived servers)."""

from __future__ import annotations

from scripts.portfolio_proof import dpe_policy_from_uplift


def test_dpe_policy_drops_surcharge_only_at_the_chosen_threshold():
    pol = dpe_policy_from_uplift(
        -0.20,
        threshold=0.05,
        ranking_supports_decision=True,
        score_threshold=0.15,
    )
    assert pol["causal_override"] is True
    assert pol["causal_recommended_treatment"] == "NO_SURCHARGE"
    assert "base fare" in pol["pricing_action"].lower()


def test_dpe_policy_does_not_cut_without_a_confirmed_threshold():
    pol = dpe_policy_from_uplift(
        -0.20,
        threshold=0.05,
        ranking_supports_decision=True,
        score_threshold=None,
    )
    assert pol["causal_override"] is False
    assert pol["test_group_if_dpe"] == "ADDITIVE"


def test_dpe_policy_persuadable_no_override():
    pol = dpe_policy_from_uplift(0.2, threshold=0.05, ranking_supports_decision=True)
    assert pol["causal_override"] is False
    assert pol["causal_recommended_treatment"] == "SURCHARGE"
