"""Event taxonomy and rule-based detection.

A structured vocabulary of the event types whose reactions we want to remember.
Detection here is rule-based and point-in-time (read from the store as of a
knowledge time); LLM labelling of the historical corpus into this taxonomy is a
later, batch enrichment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from market_trader.collectors.congress import CONGRESS_DATASET
from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.storage.bitemporal import BitemporalStore


class EventType(StrEnum):
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    CPI_SURPRISE = "cpi_surprise"
    FED_SURPRISE = "fed_surprise"
    GUIDANCE_CHANGE = "guidance_change"
    M_AND_A = "m_and_a"
    REGULATORY_ACTION = "regulatory_action"
    GEOPOLITICAL_SHOCK = "geopolitical_shock"
    INSIDER_CLUSTER_BUY = "insider_cluster_buy"
    LEADERSHIP_CONGRESS_TRADE = "leadership_congress_trade"
    NEWS_SENTIMENT_SPIKE = "news_sentiment_spike"


@dataclass(frozen=True)
class Event:
    event_type: EventType
    entity_id: str
    event_time: datetime
    knowledge_time: datetime
    payload: dict[str, Any] = field(default_factory=dict)


def detect_events(
    store: BitemporalStore,
    as_of: datetime,
    *,
    cluster_threshold: int = 3,
    cluster_window_days: int = 30,
) -> list[Event]:
    """Detect events knowable as of ``as_of`` from the bitemporal store."""
    events: list[Event] = []
    cutoff = as_of - timedelta(days=cluster_window_days)

    # Insider cluster buy: >= threshold distinct disclosed purchases within the window.
    purchases: dict[str, list[Any]] = defaultdict(list)
    for o in store.as_of(as_of, dataset=FORM4_DATASET):
        if o.value.get("is_purchase") and o.knowledge_time > cutoff:
            purchases[o.entity_id].append(o)
    for entity_id, obs in purchases.items():
        if len(obs) >= cluster_threshold:
            events.append(
                Event(
                    event_type=EventType.INSIDER_CLUSTER_BUY,
                    entity_id=entity_id,
                    event_time=max(o.event_time for o in obs),
                    knowledge_time=max(o.knowledge_time for o in obs),
                    payload={"n_buys": len(obs)},
                )
            )

    # Leadership/committee congressional trade.
    for o in store.as_of(as_of, dataset=CONGRESS_DATASET):
        if o.metadata.get("high_signal_role"):
            events.append(
                Event(
                    event_type=EventType.LEADERSHIP_CONGRESS_TRADE,
                    entity_id=o.entity_id,
                    event_time=o.event_time,
                    knowledge_time=o.knowledge_time,
                    payload={"transaction_type": o.value.get("transaction_type")},
                )
            )

    return events
