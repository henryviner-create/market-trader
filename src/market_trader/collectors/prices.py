"""Daily price bars.

Prices are knowable at the close of their own session, so ``event_time`` and
``knowledge_time`` are both that close. Bars land in the same ``price.ohlcv``
dataset the backtester reads, so real (yfinance/Tiingo/Polygon) and synthetic
prices flow through identical point-in-time machinery. ``normalize`` is pure and
offline-tested; ``fetch`` (yfinance) is wired when scheduling collection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import day_close

_OPTIONAL_FIELDS = ("open", "high", "low", "volume", "adj_close")


class PriceBar(BaseModel):
    date: date
    symbol: str
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    adj_close: float | None = None


class PriceCollector(Collector):
    source = "price"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        bars = [b if isinstance(b, PriceBar) else PriceBar.model_validate(b) for b in raw]
        out: list[Observation] = []
        for b in bars:
            close = day_close(b.date)
            value: dict[str, Any] = {"close": float(b.close)}
            for field_name in _OPTIONAL_FIELDS:
                v = getattr(b, field_name)
                if v is not None:
                    value[field_name] = float(v)
            out.append(
                Observation(
                    source=self.source,
                    dataset=PRICE_DATASET,
                    entity_type="equity",
                    entity_id=b.symbol.upper(),
                    event_time=close,
                    knowledge_time=close,
                    value=value,
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out
