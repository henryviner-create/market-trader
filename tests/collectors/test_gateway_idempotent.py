"""Ingestion must be idempotent across every store backend."""

from __future__ import annotations

from market_trader.collectors import CongressTradesCollector, IngestionGateway
from market_trader.core.identity import deterministic_id
from market_trader.storage.bitemporal import BitemporalStore

SAMPLE = [
    {
        "representative": "Nancy Pelosi",
        "chamber": "house",
        "ticker": "NVDA",
        "transaction_type": "buy",
        "transaction_date": "2023-01-10",
        "disclosure_date": "2023-02-24",
        "amount_low": 1_000_000,
        "amount_high": 5_000_000,
        "party": "D",
        "role": "leadership",
        "owner": "spouse",
    },
    {
        "representative": "Backbench Member",
        "chamber": "house",
        "ticker": "AAPL",
        "transaction_type": "sell",
        "transaction_date": "2023-01-12",
        "disclosure_date": "2023-02-20",
        "amount_low": 1_000,
        "amount_high": 15_000,
        "party": "R",
        "role": "member",
        "owner": "self",
    },
]


def test_reingest_is_a_noop(store: BitemporalStore) -> None:
    gateway = IngestionGateway(store)
    observations = CongressTradesCollector().normalize(SAMPLE)

    first = gateway.ingest(observations)
    second = gateway.ingest(CongressTradesCollector().normalize(SAMPLE))

    assert first.newly_added == len(SAMPLE)
    assert second.newly_added == 0  # nothing new the second time
    assert store.count() == len(SAMPLE)  # no duplicates


def test_deterministic_id_is_stable_across_runs() -> None:
    ids_a = [deterministic_id(o) for o in CongressTradesCollector().normalize(SAMPLE)]
    ids_b = [deterministic_id(o) for o in CongressTradesCollector().normalize(SAMPLE)]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(SAMPLE)  # distinct facts get distinct ids
