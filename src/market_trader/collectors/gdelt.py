"""GDELT global news.

News is knowable when published, so ``event_time`` and ``knowledge_time`` are both
the article's seen-date. Articles are entity-linked to a ticker where possible
(otherwise filed under a global bucket). Tone is carried for the sentiment family;
ten sources reporting one event is still one event — novelty/dedup is handled in
the signal tier, not here.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close

NEWS_DATASET = "news.article"


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
                    event_time=seen,
                    knowledge_time=seen,
                    value={"title": a.title, "url": a.url, "source": a.source_name, "tone": a.tone},
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out
