"""The LLM breadth factory end to end: news -> Opus extract -> store -> Feature (no lookahead)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_trader.collectors.gdelt import GdeltNewsCollector, NewsArticle
from market_trader.collectors.llm_features import (
    LLM_SIGNAL_DATASET,
    LLMSignalCollector,
    extract_signals_for_universe,
    headlines_by_symbol,
)
from market_trader.features.llm import LLMNewsSentiment
from market_trader.reasoning.extraction import ExtractedSignal
from market_trader.reasoning.llm import MockLLMProvider
from market_trader.storage import InMemoryBitemporalStore

AS_OF = datetime(2024, 6, 10, 12, tzinfo=UTC)


def _store_with_news() -> InMemoryBitemporalStore:
    arts = [
        NewsArticle(
            seendate=(AS_OF - timedelta(days=1)).date(), title="ABC wins contract", symbol="ABC"
        ),
        NewsArticle(
            seendate=(AS_OF - timedelta(days=2)).date(), title="ABC expands abroad", symbol="ABC"
        ),
        NewsArticle(
            seendate=(AS_OF - timedelta(days=1)).date(), title="XYZ faces a probe", symbol="XYZ"
        ),
    ]
    store = InMemoryBitemporalStore()
    store.add_many(GdeltNewsCollector().normalize(arts))
    return store


def test_headlines_grouped_by_symbol_within_window() -> None:
    heads = headlines_by_symbol(_store_with_news(), AS_OF, ["ABC", "XYZ", "NONE"], window_days=7)
    assert set(heads) == {"ABC", "XYZ"}  # NONE has no news
    assert len(heads["ABC"]) == 2


def test_collector_normalizes_to_llm_signal_dataset() -> None:
    kt = datetime(2024, 6, 10, tzinfo=UTC)
    obs = LLMSignalCollector().normalize([(ExtractedSignal("abc", 0.5, 0.9, "r"), kt)])
    assert len(obs) == 1 and obs[0].dataset == LLM_SIGNAL_DATASET
    assert obs[0].entity_id == "ABC" and obs[0].knowledge_time == kt
    assert obs[0].value["score"] == 0.5 and obs[0].value["confidence"] == 0.9


def test_extract_normalize_and_feature_roundtrip() -> None:
    store = _store_with_news()
    provider = MockLLMProvider('{"score": 0.8, "confidence": 0.5, "rationale": "positive"}')

    extracted = extract_signals_for_universe(store, provider, ["ABC", "XYZ"], AS_OF, window_days=7)
    assert len(extracted) == 2  # both names have news to read
    store.add_many(LLMSignalCollector().normalize(extracted))

    feat = LLMNewsSentiment(window_days=10).compute(
        store, AS_OF + timedelta(hours=1), ["ABC", "XYZ", "NONE"]
    )
    assert abs(feat["ABC"] - 0.4) < 1e-9  # score 0.8 * confidence 0.5
    assert abs(feat["XYZ"] - 0.4) < 1e-9
    assert feat["NONE"] == 0.0  # no extracted signal -> neutral


def test_feature_has_no_lookahead() -> None:
    store = _store_with_news()
    provider = MockLLMProvider('{"score": 1.0, "confidence": 1.0}')
    store.add_many(
        LLMSignalCollector().normalize(
            extract_signals_for_universe(store, provider, ["ABC"], AS_OF, window_days=7)
        )
    )
    # The signal's knowledge_time is AS_OF; a day earlier it must not be visible.
    earlier = LLMNewsSentiment().compute(store, AS_OF - timedelta(days=1), ["ABC"])
    assert earlier["ABC"] == 0.0
