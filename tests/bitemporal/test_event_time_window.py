"""The event-time ``since`` window: load only recent history (the deep-backfill perf guard).

A 5-year backfill is ~300k rows; the diagnostics only need the last year or two, so the SQL
store can floor the load by ``event_time`` instead of deserializing the whole table.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from market_trader.core.schema import Observation
from market_trader.core.time import DISTANT_FUTURE
from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore


def _sqlite_store() -> SqlAlchemyBitemporalStore:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    store = SqlAlchemyBitemporalStore(engine)
    store.create_schema()
    return store


def _obs(entity: str, year: int) -> Observation:
    t = datetime(year, 1, 2, tzinfo=UTC)
    return Observation(
        source="price",
        dataset="price.ohlcv",
        entity_type="equity",
        entity_id=entity,
        event_time=t,
        knowledge_time=t,
        value={"close": 1.0},
    )


def test_since_loads_only_recent_event_times() -> None:
    store = _sqlite_store()
    store.add_many([_obs("OLD", 2019), _obs("MID", 2021), _obs("NEW", 2023)])

    assert len(store.as_of(DISTANT_FUTURE)) == 3  # no window: everything knowable

    recent = store.as_of(DISTANT_FUTURE, since=datetime(2022, 1, 1, tzinfo=UTC))
    assert {o.entity_id for o in recent} == {"NEW"}  # the deep-backfill rows are left on disk

    everything = store.as_of(DISTANT_FUTURE, since=datetime(2000, 1, 1, tzinfo=UTC))
    assert len(everything) == 3  # a floor below all event times is a no-op
