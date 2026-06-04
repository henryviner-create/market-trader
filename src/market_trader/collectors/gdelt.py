"""GDELT global news.

News is knowable when published, so ``event_time`` and ``knowledge_time`` are both
the article's seen-date. Articles are entity-linked to a ticker where possible
(otherwise filed under a global bucket). Tone is carried for the sentiment family;
ten sources reporting one event is still one event — novelty/dedup is handled in
the signal tier, not here.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close
from market_trader.observability import get_logger

NEWS_DATASET = "news.article"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_log = get_logger("gdelt")


class NewsArticle(BaseModel):
    seendate: date
    title: str
    url: str | None = None
    source_name: str | None = None
    tone: float | None = None
    symbol: str | None = None  # entity-linked ticker, if resolved


class GdeltNewsCollector(Collector):
    source = "gdelt"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        articles = [a if isinstance(a, NewsArticle) else NewsArticle.model_validate(a) for a in raw]
        out: list[Observation] = []
        for a in articles:
            seen = day_close(a.seendate)
            linked = a.symbol is not None and a.symbol.strip() != ""
            out.append(
                Observation(
                    source=self.source,
                    dataset=NEWS_DATASET,
                    entity_type="equity" if linked else "news_global",
                    entity_id=a.symbol.upper() if linked and a.symbol else "GLOBAL",
                    ref=a.url or a.title,
                    event_time=seen,
                    knowledge_time=seen,
                    value={"title": a.title, "url": a.url, "source": a.source_name, "tone": a.tone},
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out


# (url) -> parsed JSON payload
NewsTransport = Callable[[str], dict[str, Any]]


def _gdelt_get(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "market-trader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # fixed https host
        raw = resp.read()
    return json.loads(raw) if raw else {}


def _parse_seendate(raw: Any) -> date:
    s = str(raw)
    return (
        datetime.strptime(s[:8], "%Y%m%d").date()
        if len(s) >= 8 and s[:8].isdigit()
        else date.today()
    )


class GdeltClient:
    """Fetch recent articles from the free GDELT 2.0 DOC API (no key required).

    ArtList carries reliable news *flow* (volume/attention); per-article tone is
    only present for some sources, so sentiment is best-effort and a richer paid
    feed can be swapped in later. Entity-linking here is by query string.
    """

    def __init__(
        self,
        *,
        base_url: str = GDELT_DOC_API,
        transport: NewsTransport | None = None,
        max_records: int = 50,
        timeout_seconds: float = 10.0,
        budget_seconds: float = 45.0,
    ) -> None:
        self._base = base_url
        self._get = transport or (lambda url: _gdelt_get(url, timeout=timeout_seconds))
        self._max = max_records
        self._budget = budget_seconds

    def fetch_articles(
        self, query: str, *, symbol: str | None = None, timespan: str = "3d"
    ) -> list[NewsArticle]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": self._max,
                "timespan": timespan,
                "sort": "DateDesc",
            }
        )
        payload = self._get(f"{self._base}?{params}")
        return [
            NewsArticle(
                seendate=_parse_seendate(a.get("seendate")),
                title=str(a.get("title", "")),
                url=a.get("url"),
                source_name=a.get("domain"),
                tone=a.get("tone"),
                symbol=symbol,
            )
            for a in (payload.get("articles") or [])
        ]

    def fetch_for_symbols(
        self, symbols: Sequence[str], *, timespan: str = "3d"
    ) -> list[NewsArticle]:
        """Best-effort per-symbol fetch, bounded by a wall-clock budget.

        One symbol erroring never aborts the batch. The budget is the load-bearing
        part: GDELT's free API throttles, and a per-symbol sweep over a ~140-name
        universe could otherwise stall a whole trading cycle for minutes. Once the
        budget elapses the sweep stops and the remaining names are skipped this
        cycle — news is a best-effort overlay, never on the critical path.
        """
        out: list[NewsArticle] = []
        start = time.monotonic()
        fetched = 0
        for s in symbols:
            if time.monotonic() - start > self._budget:
                _log.warning(
                    "gdelt_budget_exceeded",
                    fetched=fetched,
                    total=len(symbols),
                    budget_seconds=self._budget,
                )
                break
            try:
                out.extend(self.fetch_articles(s, symbol=s, timespan=timespan))
                fetched += 1
            except Exception:  # best-effort batch: skip a symbol that errors
                continue
        _log.info(
            "gdelt_fetch",
            symbols=fetched,
            articles=len(out),
            elapsed_s=round(time.monotonic() - start, 1),
        )
        return out
