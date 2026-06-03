"""Event taxonomy detection and surprise encoding."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import CongressTradesCollector, Form4Collector, IngestionGateway
from market_trader.core.time import day_close
from market_trader.memory import EventType, detect_events, encode_surprise
from market_trader.storage import InMemoryBitemporalStore


def test_encode_surprise_sign_and_standardization() -> None:
    beat = encode_surprise(1.05, 1.00)
    assert abs(beat.surprise - 0.05) < 1e-9
    assert beat.direction == 1 and beat.standardized is None

    miss = encode_surprise(0.90, 1.00, dispersion=0.05)
    assert miss.direction == -1
    assert abs((miss.standardized or 0.0) + 2.0) < 1e-9


def test_detects_insider_cluster_and_leadership_trade() -> None:
    store = InMemoryBitemporalStore()
    gateway = IngestionGateway(store)
    # Three *different* insiders buying the same name within the window.
    gateway.ingest(
        Form4Collector().normalize(
            [
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": name,
                    "transaction_code": "P",
                    "transaction_date": "2023-01-10",
                    "filing_date": "2023-01-12",
                }
                for name in ("Alice", "Bob", "Carol")
            ]
        )
    )
    gateway.ingest(
        CongressTradesCollector().normalize(
            [
                {
                    "representative": "Leader",
                    "chamber": "house",
                    "ticker": "NVDA",
                    "transaction_type": "buy",
                    "transaction_date": "2023-01-01",
                    "disclosure_date": "2023-02-15",
                    "role": "leadership",
                }
            ]
        )
    )
    assert store.count() == 4  # ref discriminator keeps the 3 insiders distinct

    events = detect_events(
        store, day_close(date(2023, 2, 20)), cluster_threshold=3, cluster_window_days=60
    )
    found = {(e.event_type, e.entity_id) for e in events}
    assert (EventType.INSIDER_CLUSTER_BUY, "AAPL") in found
    assert (EventType.LEADERSHIP_CONGRESS_TRADE, "NVDA") in found
