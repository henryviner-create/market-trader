"""The dashboard data layer renders from the store, respecting knowledge time."""

from __future__ import annotations

from datetime import date

from market_trader.collectors import (
    CongressTradesCollector,
    Form4Collector,
    FredSeriesCollector,
    GdeltNewsCollector,
    IngestionGateway,
    PriceCollector,
)
from market_trader.core.time import day_close
from market_trader.presentation import build_dashboard_data
from market_trader.storage import InMemoryBitemporalStore


def _populated_store() -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    gateway = IngestionGateway(store)
    gateway.ingest(
        PriceCollector().normalize([{"date": "2023-02-10", "symbol": "AAPL", "close": 150.0}])
    )
    gateway.ingest(
        Form4Collector().normalize(
            [
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": "Insider",
                    "transaction_code": "P",
                    "transaction_date": "2023-01-20",
                    "filing_date": "2023-01-22",
                }
            ]
        )
    )
    gateway.ingest(
        CongressTradesCollector().normalize(
            [
                {
                    "representative": "Member",
                    "chamber": "house",
                    "ticker": "AAPL",
                    "transaction_type": "buy",
                    "transaction_date": "2022-12-01",
                    "disclosure_date": "2023-01-15",
                    "role": "leadership",
                }
            ]
        )
    )
    gateway.ingest(
        GdeltNewsCollector().normalize(
            [{"seendate": "2023-02-09", "title": "Apple news", "symbol": "AAPL"}]
        )
    )
    gateway.ingest(
        FredSeriesCollector("DGS10").normalize(
            [{"date": "2023-01-01", "realtime_start": "2023-02-01", "value": "3.5"}]
        )
    )
    return store


def test_dashboard_reflects_what_was_knowable() -> None:
    store = _populated_store()
    data = build_dashboard_data(store, day_close(date(2023, 2, 15)))

    assert "AAPL" in data.watchlist
    assert data.latest_prices.get("AAPL") == 150.0
    assert data.macro.get("DGS10") == 3.5
    assert len(data.recent_insider) == 1 and data.recent_insider[0]["is_purchase"] is True
    assert len(data.recent_congress) == 1
    assert len(data.recent_news) == 1


def test_dashboard_hides_not_yet_disclosed_congress_trade() -> None:
    store = _populated_store()
    # Congress trade disclosed 2023-01-15; as of 2023-01-10 it is not yet knowable.
    early = build_dashboard_data(store, day_close(date(2023, 1, 10)))
    assert early.recent_congress == []
    # ...but the insider purchase (filed 2023-01-22) is also still hidden then.
    assert early.recent_insider == []
