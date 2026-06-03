"""The load-bearing test of the whole system.

Property: for *any* set of observations and *any* query knowledge time ``K``,
``store.as_of(K)`` returns exactly the facts with ``knowledge_time <= K`` — never
one knowable only later. If this holds, lookahead bias cannot enter through the
storage tier.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from market_trader.core.schema import Observation
from market_trader.storage.bitemporal import BitemporalStore, collapse_latest_revisions

_SOURCES = ["edgar", "fred", "price", "congress"]
_DATASETS = ["price.ohlcv", "filing.form4", "macro.series"]
_ENTITY_TYPES = ["equity", "macro_series", "person"]
_ENTITIES = ["AAPL", "MSFT", "DGS10", "PELOSI"]

_times = st.datetimes(
    min_value=datetime(2018, 1, 1),
    max_value=datetime(2024, 1, 1),
).map(lambda d: d.replace(tzinfo=UTC))


@st.composite
def _observations(draw: st.DrawFn) -> Observation:
    return Observation(
        source=draw(st.sampled_from(_SOURCES)),
        dataset=draw(st.sampled_from(_DATASETS)),
        entity_type=draw(st.sampled_from(_ENTITY_TYPES)),
        entity_id=draw(st.sampled_from(_ENTITIES)),
        event_time=draw(_times),
        knowledge_time=draw(_times),
        value={"x": draw(st.integers(min_value=-5, max_value=5))},
        revision=draw(st.integers(min_value=0, max_value=3)),
    )


@pytest.mark.parametrize("kind", ["inmemory", "sqlite"])
@settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(observations=st.lists(_observations(), max_size=40), query=_times)
def test_as_of_never_reveals_the_future(
    kind: str,
    store_factory: Callable[[str], BitemporalStore],
    observations: list[Observation],
    query: datetime,
) -> None:
    store = store_factory(kind)
    store.add_many(observations)

    # 1) The core guarantee: nothing knowable only after `query` is returned.
    raw = store.as_of(query, latest_revision_only=False)
    assert all(o.knowledge_time <= query for o in raw)

    # 2) Completeness, checked against an independent filter (not the store's code).
    expected_ids = {o.observation_id for o in observations if o.knowledge_time <= query}
    assert {o.observation_id for o in raw} == expected_ids

    # 3) Revision collapsing also respects the horizon and matches the shared oracle.
    collapsed = store.as_of(query, latest_revision_only=True)
    assert all(o.knowledge_time <= query for o in collapsed)
    visible = [o for o in observations if o.knowledge_time <= query]
    expected_collapsed = {o.observation_id for o in collapse_latest_revisions(visible)}
    assert {o.observation_id for o in collapsed} == expected_collapsed
