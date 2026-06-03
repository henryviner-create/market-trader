"""FRED macro series, with vintage-correct knowledge times.

FRED/ALFRED expose *vintages*: each observation carries a ``realtime_start`` —
the date that value first became known (revisions get later vintages). That field
is exactly our ``knowledge_time``, so macro data is honestly point-in-time
without any guesswork. ``event_time`` is the period the value describes.

``normalize`` parses the FRED ``series/observations`` JSON (or a list of the same
records). Live ``fetch`` against the FRED API (which needs ``MT_FRED_API_KEY`` and
``realtime_start=`` for vintages) is added when wiring scheduled collection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, field_validator

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close

FRED_DATASET = "macro.series"


class FredObservation(BaseModel):
    date: date
    realtime_start: date
    value: float | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _blank_is_missing(cls, v: Any) -> Any:
        if v in (".", "", None):  # FRED encodes missing values as "."
            return None
        return v


class FredSeriesCollector(Collector):
    source = "fred"
    parser_version = 1

    def __init__(self, series_id: str) -> None:
        self.series_id = series_id.upper()

    def normalize(self, raw: Any) -> list[Observation]:
        records = raw["observations"] if isinstance(raw, dict) else raw
        out: list[Observation] = []
        for rec in records:
            o = rec if isinstance(rec, FredObservation) else FredObservation.model_validate(rec)
            if o.value is None:
                continue
            out.append(
                Observation(
                    source=self.source,
                    dataset=FRED_DATASET,
                    entity_type="macro_series",
                    entity_id=self.series_id,
                    event_time=day_close(o.date),
                    knowledge_time=day_close(o.realtime_start),  # vintage = when value was known
                    value={"value": float(o.value), "series_id": self.series_id},
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out
