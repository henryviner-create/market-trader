"""The pluggable news feed: the GDELT adapter normalizes to news.article obs."""

from __future__ import annotations

from market_trader.collectors.gdelt import NEWS_DATASET, GdeltClient
from market_trader.collectors.news_feed import GdeltNewsFeed, NewsFeed


def test_gdelt_feed_normalizes_and_satisfies_protocol() -> None:
    payload = {
        "articles": [
            {
                "seendate": "20260601T120000Z",
                "title": "ACME wins big",
                "url": "http://a/1",
                "domain": "reuters.com",
                "tone": 3.4,
            }
        ]
    }
    feed = GdeltNewsFeed(GdeltClient(transport=lambda _url: payload))
    obs = feed.fetch_recent(["ACME"], lookback_minutes=120)

    assert isinstance(feed, NewsFeed)  # structural Protocol — a paid feed can replace it
    assert len(obs) == 1
    assert obs[0].dataset == NEWS_DATASET and obs[0].entity_id == "ACME"
    assert obs[0].value["tone"] == 3.4 and obs[0].value["url"] == "http://a/1"


def test_gdelt_feed_empty_payload_is_safe() -> None:
    feed = GdeltNewsFeed(GdeltClient(transport=lambda _url: {}))
    assert feed.fetch_recent(["ACME"], lookback_minutes=60) == []
