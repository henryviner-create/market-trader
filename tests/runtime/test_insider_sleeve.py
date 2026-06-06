"""Insider-cluster sleeve execution: open a fresh gated cluster, then close it at horizon."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.config import Settings
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.execution.broker import OrderSide
from market_trader.execution.paper_broker import PaperBroker
from market_trader.memory.event_study import EventOutcomeDistribution
from market_trader.runtime.insider_events import INSIDER_CLUSTER
from market_trader.runtime.insider_sleeve import (
    active_insider_positions,
    run_insider_sleeve_cycle,
)
from market_trader.storage import InMemoryBitemporalStore

AS_OF = datetime(2024, 6, 10, 12, tzinfo=UTC)
GATE = {
    INSIDER_CLUSTER: EventOutcomeDistribution(
        "insider_cluster_buy", 327, 0.03, 0.13, 3.98, 0.63, -0.05, 0.02, 0.12
    )
}
SETTINGS = Settings(
    execution_mode="paper",
    universe="ABC,S1,S2",
    capital_ceiling=100_000.0,
    insider_sleeve_budget=0.10,
    insider_sleeve_max_names=5,
    insider_sleeve_hold_days=5,
)


def _store_with_cluster() -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    store.add_many(
        synthetic_price_observations(symbols=["ABC", "S1", "S2"], start=date(2024, 5, 1), n_days=30)
    )
    for j, d in enumerate((1, 2, 3)):  # 3 fresh insider purchases for ABC -> a cluster
        kt = AS_OF - timedelta(days=d)
        store.add_many(
            [
                Observation(
                    source="edgar",
                    dataset=FORM4_DATASET,
                    entity_type="equity",
                    entity_id="ABC",
                    event_time=kt,
                    knowledge_time=kt,
                    value={"is_purchase": True, "insider_name": f"insider-{j}"},
                    metadata={},
                )
            ]
        )
    return store


def _broker(store: InMemoryBitemporalStore, as_of: datetime) -> PaperBroker:
    prices = {
        o.entity_id: float(o.value["close"])
        for o in store.as_of(as_of, dataset=PRICE_DATASET)
        if o.entity_id in {"ABC", "S1", "S2"}
    }
    return PaperBroker(prices, starting_cash=100_000.0)


def test_opens_a_fresh_gated_cluster_then_closes_at_horizon() -> None:
    store = _store_with_cluster()
    broker = _broker(store, AS_OF)

    opened = run_insider_sleeve_cycle(SETTINGS, store=store, broker=broker, as_of=AS_OF, gate=GATE)
    assert opened.opened == ["ABC"]
    assert any(o.symbol == "ABC" and o.side == OrderSide.BUY for o in opened.orders)
    assert "ABC" in active_insider_positions(store, AS_OF)  # persisted as open

    # A pass past the time-box exits the position (same broker still holds ABC).
    later = AS_OF + timedelta(days=6)
    done = run_insider_sleeve_cycle(SETTINGS, store=store, broker=broker, as_of=later, gate=GATE)
    assert done.closed == ["ABC"]
    assert any(o.symbol == "ABC" and o.side == OrderSide.SELL for o in done.orders)
    assert "ABC" not in active_insider_positions(store, later)  # back to flat


def test_does_not_trade_until_the_gate_is_cleared() -> None:
    store = _store_with_cluster()
    broker = _broker(store, AS_OF)
    result = run_insider_sleeve_cycle(SETTINGS, store=store, broker=broker, as_of=AS_OF, gate={})
    assert result.opened == [] and result.orders == []  # event type not significant -> flat
