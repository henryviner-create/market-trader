"""Pluggable news feed for the event-driven sleeve.

A thin seam over a news source: :meth:`NewsFeed.fetch_recent` returns normalised
``news.article`` observations for a watchlist. The GDELT adapter reuses the
existing free client/collector; a real-time *paid* provider implements the same
Protocol and drops in with no other code changes.

GDELT updates on a ~15-minute cycle with lag, so a sleeve built on it trades the
*drift after* news is digested (a documented, retail-tradeable effect), not the
instantaneous move. Swapping a low-latency feed in later is the only way closer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from market_trader.collectors.gdelt import GdeltClient, GdeltNewsCollector
from market_trader.core.schema import Observation


@runtime_checkable
class NewsFeed(Protocol):
    def fetch_recent(self, symbols: Sequence[str], *, lookback_minutes: int) -> list[Observation]:
        """Recent news for ``symbols`` as normalised ``news.article`` observations."""
        ...


def _timespan_for(lookback_minutes: int) -> str:
    # GDELT's `timespan` accepts e.g. "120min"; never poll a window shorter than
    # one update cycle, or fresh stories are missed.
    return f"{max(15, int(lookback_minutes))}min"


class GdeltNewsFeed:
    """:class:`NewsFeed` backed by the free GDELT DOC API (reuses GdeltClient)."""

    def __init__(self, client: GdeltClient | None = None) -> None:
        self._client = client or GdeltClient()
        self._collector = GdeltNewsCollector()

    def fetch_recent(self, symbols: Sequence[str], *, lookback_minutes: int) -> list[Observation]:
        articles = self._client.fetch_for_symbols(symbols, timespan=_timespan_for(lookback_minutes))
        return self._collector.normalize(articles)
