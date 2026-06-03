"""Congressional disclosures: correct lag stamping, and invisible until disclosed."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import CongressTradesCollector, IngestionGateway
from market_trader.collectors.congress import CONGRESS_DATASET
from market_trader.core.time import day_close
from market_trader.storage import InMemoryBitemporalStore

PELOSI_NVDA = {
    "representative": "Nancy Pelosi",
    "chamber": "house",
    "ticker": "nvda",
    "transaction_type": "buy",
    "transaction_date": "2023-01-10",
    "disclosure_date": "2023-02-24",  # 45-day lag
    "amount_low": 1_000_000,
    "amount_high": 5_000_000,
    "party": "D",
    "role": "leadership",
    "owner": "spouse",
}


def test_normalize_stamps_event_and_knowledge_times() -> None:
    (obs,) = CongressTradesCollector().normalize([PELOSI_NVDA])
    assert obs.entity_id == "NVDA"  # uppercased
    assert obs.event_time.date() == date(2023, 1, 10)  # transaction date
    assert obs.knowledge_time.date() == date(2023, 2, 24)  # disclosure date
    assert obs.metadata["disclosure_lag_days"] == 45
    assert obs.metadata["high_signal_role"] is True  # leadership
    assert obs.value["transaction_type"] == "buy"


def test_trade_is_invisible_until_its_disclosure_date() -> None:
    store = InMemoryBitemporalStore()
    IngestionGateway(store).ingest(CongressTradesCollector().normalize([PELOSI_NVDA]))

    # Before disclosure: the trade existed in the world but was not knowable to us.
    before = store.as_of(day_close(date(2023, 1, 20)), dataset=CONGRESS_DATASET)
    assert before == []

    # On/after disclosure: now visible.
    after = store.as_of(day_close(date(2023, 2, 25)), dataset=CONGRESS_DATASET)
    assert len(after) == 1
    assert after[0].entity_id == "NVDA"
