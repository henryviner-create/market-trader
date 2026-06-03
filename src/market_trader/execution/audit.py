"""Execution audit log.

Every signal-to-fill action is persisted as a bitemporal observation — for the
feedback loop and for diagnosing the inevitable first incident. Idempotent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.storage.bitemporal import BitemporalStore

EXECUTION_DATASET = "execution.audit"


def log_execution_event(
    store: BitemporalStore,
    *,
    as_of: datetime,
    symbol: str,
    event: str,
    detail: dict[str, Any],
) -> None:
    observation = with_deterministic_id(
        Observation(
            source="execution",
            dataset=EXECUTION_DATASET,
            entity_type="equity",
            entity_id=symbol,
            ref=f"{event}:{detail.get('client_order_id', '')}",
            event_time=as_of,
            knowledge_time=as_of,
            value={"event": event, **{k: v for k, v in detail.items()}},
        )
    )
    store.upsert_many([observation])


def load_execution_audit(store: BitemporalStore, as_of: datetime) -> list[Observation]:
    return store.as_of(as_of, dataset=EXECUTION_DATASET)
