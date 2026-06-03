"""Alpaca market-data client: request shape + PriceBar-compatible records (offline).

The client takes an injectable transport, so we exercise the URL it builds and the
records it returns with zero network — then prove those records flow cleanly through
the same ``PriceCollector`` normalisation that synthetic and yfinance bars use.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from market_trader.collectors import PriceCollector
from market_trader.collectors.alpaca import AlpacaDataClient, AlpacaDataError
from market_trader.core.synthetic import PRICE_DATASET

_PAYLOAD: dict[str, Any] = {
    "bars": {
        "AAPL": [
            {
                "t": "2023-06-01T04:00:00Z",
                "o": 180.0,
                "h": 182.0,
                "l": 179.0,
                "c": 181.0,
                "v": 1000,
            },
            {
                "t": "2023-06-02T04:00:00Z",
                "o": 181.0,
                "h": 184.0,
                "l": 180.5,
                "c": 183.5,
                "v": 1200,
            },
        ],
        "MSFT": [
            {"t": "2023-06-01T04:00:00Z", "o": 330.0, "h": 333.0, "l": 329.0, "c": 332.0, "v": 800},
        ],
    }
}


def _transport(status: int = 200, payload: dict[str, Any] | None = None):
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return status, (_PAYLOAD if payload is None else payload)

    return transport, calls


def test_requires_keys() -> None:
    with pytest.raises(AlpacaDataError):
        AlpacaDataClient("", "")


def test_fetch_daily_bars_shape_and_query() -> None:
    transport, calls = _transport()
    client = AlpacaDataClient("k", "s", transport=transport)
    records = client.fetch_daily_bars(
        ["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 3)
    )

    assert len(records) == 3  # 2 AAPL + 1 MSFT
    assert "symbols=AAPL%2CMSFT" in calls[0]  # comma-joined + URL-encoded
    assert "timeframe=1Day" in calls[0]
    assert "feed=iex" in calls[0]  # free feed by default; sip 403s on free plans

    aapl = next(r for r in records if r["symbol"] == "AAPL")
    assert aapl["date"] == "2023-06-01"
    assert aapl["close"] == 181.0 and aapl["volume"] == 1000
    assert {"date", "symbol", "open", "high", "low", "close", "volume"} <= set(aapl)


def test_records_flow_through_price_collector() -> None:
    transport, _ = _transport()
    client = AlpacaDataClient("k", "s", transport=transport)
    records = client.fetch_daily_bars(
        ["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 3)
    )

    observations = PriceCollector().normalize(records)  # same path as synthetic/yfinance
    assert observations
    assert all(o.dataset == PRICE_DATASET for o in observations)
    aapl = [o for o in observations if o.entity_id == "AAPL"]
    assert len(aapl) == 2
    assert all(o.knowledge_time == o.event_time for o in aapl)  # known at the close


def test_feed_is_overridable() -> None:
    transport, calls = _transport()
    client = AlpacaDataClient("k", "s", transport=transport)
    client.fetch_daily_bars(["AAPL"], start=date(2023, 6, 1), end=date(2023, 6, 3), feed="sip")
    assert "feed=sip" in calls[0]  # paid plans can opt back into the consolidated feed


def test_fetch_daily_bars_follows_pagination() -> None:
    # A broad universe (many symbols x history) spans multiple pages; the client
    # must follow next_page_token or it would silently drop the later symbols.
    pages: list[dict[str, Any]] = [
        {"bars": {"AAPL": [{"t": "2023-06-01T04:00:00Z", "c": 181.0}]}, "next_page_token": "P2"},
        {"bars": {"MSFT": [{"t": "2023-06-01T04:00:00Z", "c": 332.0}]}, "next_page_token": None},
    ]
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return 200, pages[len(calls) - 1]

    client = AlpacaDataClient("k", "s", transport=transport)
    records = client.fetch_daily_bars(
        ["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 2)
    )

    assert len(calls) == 2  # followed the page token
    assert "page_token=P2" in calls[1]
    assert {r["symbol"] for r in records} == {"AAPL", "MSFT"}


def test_http_error_status_raises() -> None:
    transport, _ = _transport(status=403, payload={"message": "forbidden"})
    client = AlpacaDataClient("k", "s", transport=transport)
    with pytest.raises(AlpacaDataError):
        client.fetch_daily_bars(["AAPL"], start=date(2023, 6, 1), end=date(2023, 6, 3))
