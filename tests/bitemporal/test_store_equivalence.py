"""Backends must agree, and corrections must respect knowledge time.

These run against in-memory, SQLite, and (in CI) Postgres via the ``store``
fixture, so the SQL store is validated against the in-memory oracle on real
Postgres too.
"""

from __future__ import annotations

from datetime import date

from market_trader.core.schema import Observation
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.core.time import day_close
from market_trader.storage import InMemoryBitemporalStore
from market_trader.storage.bitemporal import BitemporalStore


def test_store_matches_in_memory_oracle(store: BitemporalStore) -> None:
    observations = synthetic_price_observations(
        symbols=["A", "B", "C"], start=date(2022, 1, 3), n_days=30, seed=3
    )
    oracle = InMemoryBitemporalStore()
    oracle.add_many(observations)
    store.add_many(observations)

    for k in (
        day_close(date(2022, 1, 5)),
        day_close(date(2022, 1, 20)),
        day_close(date(2022, 2, 10)),
    ):
        got = store.as_of(k, dataset="price.ohlcv")
        want = oracle.as_of(k, dataset="price.ohlcv")
        assert [o.observation_id for o in got] == [o.observation_id for o in want]
        assert [round(o.value["close"], 9) for o in got] == [
            round(o.value["close"], 9) for o in want
        ]


def test_correction_is_invisible_until_its_knowledge_time(store: BitemporalStore) -> None:
    event = day_close(date(2022, 1, 10))
    original = Observation(
        source="price",
        dataset="price.ohlcv",
        entity_type="equity",
        entity_id="A",
        event_time=event,
        knowledge_time=event,
        value={"close": 100.0},
        revision=0,
    )
    correction = Observation(
        source="price",
        dataset="price.ohlcv",
        entity_type="equity",
        entity_id="A",
        event_time=event,
        knowledge_time=day_close(date(2022, 2, 1)),
        value={"close": 105.0},
        revision=1,
    )
    store.add_many([original, correction])

    before = store.as_of(day_close(date(2022, 1, 20)), entity_id="A")
    assert len(before) == 1
    assert before[0].value["close"] == 100.0  # pre-correction view

    after = store.as_of(day_close(date(2022, 2, 5)), entity_id="A")
    assert len(after) == 1
    assert after[0].value["close"] == 105.0  # correction now knowable
