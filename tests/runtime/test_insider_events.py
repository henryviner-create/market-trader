"""The insider-cluster sleeve decision: gated, fresh-only, time-boxed, deduped."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.core.schema import Observation
from market_trader.memory.event_study import EventOutcomeDistribution
from market_trader.runtime.insider_events import INSIDER_CLUSTER, insider_cluster_entries
from market_trader.storage import InMemoryBitemporalStore

AS_OF = datetime(2024, 6, 10, 12, tzinfo=UTC)
# A significant, positive gate (mirrors the live event-study: CAR +3%, t~4, n=327).
SIGNIFICANT = EventOutcomeDistribution(
    "insider_cluster_buy", 327, 0.03, 0.13, 3.98, 0.63, -0.05, 0.02, 0.12
)
GATE = {INSIDER_CLUSTER: SIGNIFICANT}


def _cluster_store(*clusters: tuple[str, list[datetime]]) -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    obs = [
        Observation(
            source="edgar",
            dataset=FORM4_DATASET,
            entity_type="equity",
            entity_id=symbol,
            event_time=kt,
            knowledge_time=kt,
            value={"is_purchase": True, "insider_name": f"{symbol}-insider-{j}"},
            metadata={},
        )
        for symbol, kts in clusters
        for j, kt in enumerate(kts)
    ]
    store.add_many(obs)
    return store


def _recent(*offsets: int) -> list[datetime]:
    return [AS_OF - timedelta(days=d) for d in offsets]


def test_opens_a_fresh_gated_insider_cluster() -> None:
    store = _cluster_store(("ABC", _recent(1, 2, 3)))
    entries = insider_cluster_entries(store, AS_OF, gate=GATE, hold_days=5)
    assert len(entries) == 1
    e = entries[0]
    assert e.symbol == "ABC" and e.n_buys == 3
    assert e.exit_by == AS_OF + timedelta(days=5)  # time-boxed
    assert e.expected_car == SIGNIFICANT.mean_car  # carries the gate's measured drift


def test_flat_until_the_event_type_clears_the_gate() -> None:
    store = _cluster_store(("ABC", _recent(1, 2, 3)))
    assert insider_cluster_entries(store, AS_OF, gate={}) == []  # not gated at all
    not_significant = EventOutcomeDistribution(
        "insider_cluster_buy", 5, 0.01, 0.2, 0.5, 0.5, 0, 0, 0
    )
    assert insider_cluster_entries(store, AS_OF, gate={INSIDER_CLUSTER: not_significant}) == []


def test_skips_a_stale_cluster_whose_drift_already_passed() -> None:
    store = _cluster_store(("ABC", _recent(20, 21, 22)))  # detectable but knowable 20d ago
    assert insider_cluster_entries(store, AS_OF, gate=GATE, freshness_days=5) == []


def test_skips_a_name_already_held() -> None:
    store = _cluster_store(("ABC", _recent(1, 2, 3)))
    assert insider_cluster_entries(store, AS_OF, gate=GATE, held=frozenset({"ABC"})) == []


def test_caps_at_max_names_newest_first() -> None:
    store = _cluster_store(("OLDER", _recent(4, 5, 6)), ("NEWER", _recent(1, 2, 3)))
    entries = insider_cluster_entries(store, AS_OF, gate=GATE, max_names=1)
    assert len(entries) == 1 and entries[0].symbol == "NEWER"  # newest cluster wins the slot
