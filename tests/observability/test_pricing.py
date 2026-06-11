"""Tests for cost estimation."""

from openchatbi.observability.pricing import estimate_cost


def test_known_model_cost() -> None:
    # gpt-4o: $2.5/1M input, $10/1M output → 1000 in + 500 out.
    cost = estimate_cost("gpt-4o", 1000, 500)
    assert abs(cost - (1000 / 1_000_000 * 2.5 + 500 / 1_000_000 * 10.0)) < 1e-9


def test_prefix_match_is_case_insensitive() -> None:
    # Provider-prefixed / versioned names resolve via prefix lookup.
    assert estimate_cost("GPT-4o-2024-08-06", 1000, 1000) > 0.0


def test_unknown_model_returns_zero() -> None:
    assert estimate_cost("some-local-ollama-model", 9999, 9999) == 0.0
