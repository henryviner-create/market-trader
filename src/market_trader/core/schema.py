"""The canonical, bitemporal data model.

Every input source — EDGAR filings, FRED series, prices, news, congressional
disclosures — is normalised into :class:`Observation` *before* it touches
storage. One shape downstream means one set of point-in-time rules to enforce,
and one place to enforce them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_trader.core.time import ensure_utc, utcnow

LogicalKey = tuple[str, str, str, datetime]


class Observation(BaseModel):
    """One normalised, point-in-time fact.

    The pair ``(event_time, knowledge_time)`` is the heart of the system's
    honesty:

    * ``event_time``     — when the fact was true in the world (valid time).
    * ``knowledge_time`` — when the fact first became knowable to *us*.

    We deliberately do **not** require ``knowledge_time >= event_time``:
    consensus estimates and forecasts are knowable *before* the event they
    describe, and the surprise-encoding layer (Phase 3) depends on that.

    Corrections/restatements are modelled as new observations sharing a
    :attr:`logical_key` with a higher ``revision``; a reader as-of some
    knowledge time sees the most recent revision knowable by then — never a
    correction published later.
    """

    model_config = ConfigDict(frozen=True)

    source: str  # e.g. "edgar", "fred", "price", "congress", "synthetic"
    dataset: str  # e.g. "price.ohlcv", "filing.form4", "macro.series"
    entity_type: str  # e.g. "equity", "macro_series", "person", "option_contract"
    entity_id: str  # e.g. "AAPL", "DGS10", a CIK, an OCC option symbol
    event_time: datetime
    knowledge_time: datetime
    value: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = 0
    observation_id: UUID = Field(default_factory=uuid4)
    ingested_at: datetime = Field(default_factory=utcnow)

    @field_validator("event_time", "knowledge_time", "ingested_at")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @property
    def logical_key(self) -> LogicalKey:
        """Identifies the real-world fact this row is a (possibly revised) version of."""
        return (self.source, self.dataset, self.entity_id, self.event_time)
