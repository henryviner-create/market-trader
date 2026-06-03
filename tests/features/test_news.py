"""News-flow features read the news.article dataset point-in-time."""

from __future__ import annotations

from datetime import date, datetime

from market_trader.collectors.gdelt import GdeltNewsCollector, NewsArticle
from market_trader.core.time import UTC
from market_trader.features.news import NewsAttention, NewsSentiment
from market_trader.storage import InMemoryBitemporalStore

_AS_OF = datetime(2026, 6, 4, tzinfo=UTC)


def _store(articles: list[NewsArticle]) -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    store.add_many(GdeltNewsCollector().normalize(articles))
    return store


def test_news_attention_counts_recent_coverage() -> None:
    today = date(2026, 6, 3)
    store = _store(
        [
            NewsArticle(seendate=today, title="AAPL up", symbol="AAPL", tone=2.0),
            NewsArticle(seendate=today, title="AAPL launch", symbol="AAPL", tone=4.0),
            NewsArticle(seendate=today, title="XOM news", symbol="XOM", tone=-3.0),
        ]
    )
    attn = NewsAttention(window_days=7).compute(store, _AS_OF, ["AAPL", "XOM", "MSFT"])
    assert attn["AAPL"] > attn["XOM"] > attn["MSFT"]  # 2 > 1 > 0 articles
    assert attn["MSFT"] == 0.0  # no coverage -> neutral, still ranked


def test_news_sentiment_averages_tone() -> None:
    today = date(2026, 6, 3)
    store = _store(
        [
            NewsArticle(seendate=today, title="a", symbol="AAPL", tone=2.0),
            NewsArticle(seendate=today, title="b", symbol="AAPL", tone=4.0),
            NewsArticle(seendate=today, title="c", symbol="XOM", tone=-3.0),
        ]
    )
    tone = NewsSentiment(window_days=7).compute(store, _AS_OF, ["AAPL", "XOM", "MSFT"])
    assert tone["AAPL"] == 3.0 and tone["XOM"] == -3.0  # (2+4)/2 ; single
    assert tone["MSFT"] == 0.0  # no tone -> neutral


def test_news_window_excludes_stale_articles() -> None:
    store = _store([NewsArticle(seendate=date(2026, 5, 1), title="old", symbol="AAPL", tone=5.0)])
    attn = NewsAttention(window_days=7).compute(store, _AS_OF, ["AAPL"])
    assert attn["AAPL"] == 0.0  # outside the trailing window
