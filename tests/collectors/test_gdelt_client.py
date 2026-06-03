"""GDELT client parses the DOC API article list (offline, injected transport)."""

from __future__ import annotations

from typing import Any

from market_trader.collectors.gdelt import GdeltClient


def test_fetch_articles_parses_doc_api() -> None:
    payload = {
        "articles": [
            {
                "seendate": "20260603T120000Z",
                "title": "AAPL soars",
                "url": "http://x",
                "domain": "x",
            },
            {
                "seendate": "20260602T080000Z",
                "title": "AAPL dips",
                "url": "http://y",
                "domain": "y",
            },
        ]
    }
    calls: list[str] = []

    def transport(url: str) -> dict[str, Any]:
        calls.append(url)
        return payload

    arts = GdeltClient(transport=transport).fetch_articles("AAPL", symbol="AAPL", timespan="3d")

    assert len(arts) == 2
    assert arts[0].symbol == "AAPL" and arts[0].seendate.isoformat() == "2026-06-03"
    assert "query=AAPL" in calls[0] and "mode=ArtList" in calls[0]


def test_fetch_for_symbols_is_best_effort() -> None:
    def transport(url: str) -> dict[str, Any]:
        if "BAD" in url:
            raise RuntimeError("boom")
        return {"articles": [{"seendate": "20260603T120000Z", "title": "ok"}]}

    arts = GdeltClient(transport=transport).fetch_for_symbols(["GOOD", "BAD"], timespan="1d")

    assert len(arts) == 1 and arts[0].symbol == "GOOD"  # BAD failed but didn't abort the batch
