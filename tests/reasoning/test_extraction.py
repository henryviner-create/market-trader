"""The LLM breadth-factory core: call budget + structured, fail-soft extraction."""

from __future__ import annotations

import pytest

from market_trader.reasoning.budget import BudgetedProvider, LLMBudgetExceeded
from market_trader.reasoning.extraction import (
    extract_news_signal,
    parse_signal,
)
from market_trader.reasoning.llm import MockLLMProvider


def test_budgeted_provider_allows_budget_then_fails_closed() -> None:
    p = BudgetedProvider(MockLLMProvider("ok"), budget=2)
    assert p.complete(system="s", prompt="p") == "ok" and p.remaining == 1
    assert p.complete(system="s", prompt="p") == "ok" and p.remaining == 0
    with pytest.raises(LLMBudgetExceeded):
        p.complete(system="s", prompt="p")  # third call over the budget -> fail closed
    assert p.calls == 2  # the rejected call did not reach the inner provider


def test_extract_parses_and_clips_valid_json() -> None:
    provider = MockLLMProvider('{"score": 1.7, "confidence": 1.5, "rationale": "beat + raise"}')
    sig = extract_news_signal(provider, symbol="ABC", headlines=["ABC beats and raises guidance"])
    assert sig is not None
    assert sig.symbol == "ABC"
    assert sig.score == 1.0 and sig.confidence == 1.0  # out-of-range values clipped to bounds
    assert sig.rationale == "beat + raise"


def test_parse_tolerates_code_fences_and_prose() -> None:
    text = 'Here is the result:\n```json\n{"score": -0.4, "confidence": 0.6}\n```\nDone.'
    sig = parse_signal("XYZ", text)
    assert sig is not None and sig.score == -0.4 and sig.confidence == 0.6


def test_unparseable_output_is_a_missing_signal_not_a_guess() -> None:
    assert parse_signal("XYZ", "I think it's slightly bullish, maybe?") is None
    assert parse_signal("XYZ", '{"confidence": 0.5}') is None  # missing required score
    # the extractor surfaces None (neutral), never a fabricated number
    assert extract_news_signal(MockLLMProvider("not json"), symbol="XYZ", headlines=["x"]) is None


def test_empty_headlines_makes_no_call() -> None:
    provider = MockLLMProvider('{"score": 0.5, "confidence": 0.5}')
    assert extract_news_signal(provider, symbol="ABC", headlines=["", "   "]) is None
    assert provider.calls == []  # no text to read -> no LLM spend


def test_budget_exhaustion_propagates_so_the_batch_can_stop() -> None:
    provider = BudgetedProvider(MockLLMProvider('{"score": 0.1, "confidence": 0.2}'), budget=0)
    with pytest.raises(LLMBudgetExceeded):
        extract_news_signal(provider, symbol="ABC", headlines=["a headline"])
