"""Event/threshold alerts.

Turns detected events into actionable alerts (insider cluster-buy, leadership
congressional trade, ...). Alerts are inputs to judgement; routing them to
phone/email/Slack is the Phase 7 observability concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_trader.memory import EventType, detect_events
from market_trader.storage.bitemporal import BitemporalStore


@dataclass(frozen=True)
class Alert:
    level: str  # "high" | "medium" | "low"
    kind: str
    entity_id: str
    message: str
    as_of: datetime


def generate_alerts(
    store: BitemporalStore,
    as_of: datetime,
    *,
    cluster_threshold: int = 3,
    cluster_window_days: int = 30,
) -> list[Alert]:
    alerts: list[Alert] = []
    for event in detect_events(
        store, as_of, cluster_threshold=cluster_threshold, cluster_window_days=cluster_window_days
    ):
        if event.event_type == EventType.INSIDER_CLUSTER_BUY:
            alerts.append(
                Alert(
                    "high",
                    "insider_cluster_buy",
                    event.entity_id,
                    f"{event.payload.get('n_buys')} insiders bought {event.entity_id}",
                    as_of,
                )
            )
        elif event.event_type == EventType.LEADERSHIP_CONGRESS_TRADE:
            alerts.append(
                Alert(
                    "medium",
                    "leadership_congress_trade",
                    event.entity_id,
                    f"leadership {event.payload.get('transaction_type')} in {event.entity_id}",
                    as_of,
                )
            )
    return alerts
