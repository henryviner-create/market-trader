"""Intraday data layer: minute-bar fetch (shape + pagination) and normalisation.

Like the daily client test, the transport is injected so we exercise the request
the client builds and the observations it yields with zero network — then prove
those minute observations pivot into a *minute-resolution* panel, which is what
makes the existing frequency-agnostic features behave as intraday signals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.collectors.alpaca import AlpacaDataClient
from market_trader.collectors.intraday import intraday_bars_to_observations
from market_trader.core.synthetic import PRICE_INTRADAY_DATASET

_PAGE_1: dict[str, Any] = {
    "bars": {
        "AAPL": [
            {"t": "2026-06-03T13:30:00Z", "o": 180, "h": 181, "l": 179, "c": 180.5, "v": 1000},
            {"t": "2026-06-03T13:31:00Z", "o": 180.5, "h": 182, "l": 180, "c": 181.5, "v": 1100},
        ]
    },
    "next_page_token": "PAGE2",
}
_PAGE_2: dict[str, Any] = {
    "bars": {
        "AAPL": [
            {"t": "2026-06-03T13:32:00Z", "o": 181.5, "h": 183, "l": 181, "c": 182.5, "v": 1200},
        ]
    },
    "next_page_token": None,
}


def _paged_transport(pages: list[dict[str, Any]]):
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return 200, pages[len(calls) - 1]

    return transport, calls


def test_fetch_intraday_bars_query_and_pagination() -> None:
    transport, calls = _paged_transport([_PAGE_1, _PAGE_2])
    client = AlpacaDataClient("k", "s", transport=transport)
    records = client.fetch_intraday_bars(
        ["AAPL"],
        start=datetime(2026, 6, 3, 13, 30, tzinfo=UTC),
        end=datetime(2026, 6, 3, 13, 33, tzinfo=UTC),
    )

    assert len(calls) == 2  # followed next_page_token to a second page
    assert "timeframe=1Min" in calls[0]
    assert "feed=iex" in calls[0]
    assert "page_token=PAGE2" in calls[1]
    assert len(records) == 3  # 2 from page 1 + 1 from page 2
    assert records[0]["timestamp"] == "2026-06-03T13:30:00Z"  # full minute kept, not a date
    assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"} == set(records[0])


def test_intraday_bars_become_minute_resolution_panel() -> None:
    transport, _ = _paged_transport([_PAGE_1, _PAGE_2])
    client = AlpacaDataClient("k", "s", transport=transport)
    records = client.fetch_intraday_bars(
        ["AAPL"], start=datetime(2026, 6, 3, tzinfo=UTC), end=datetime(2026, 6, 3, 14, tzinfo=UTC)
    )

    obs = intraday_bars_to_observations(records)
    assert obs and all(o.dataset == PRICE_INTRADAY_DATASET for o in obs)
    assert all(o.event_time == o.knowledge_time for o in obs)  # knowable once the minute closes

    panel = observations_to_price_frame(obs)
    idx = panel.index
    assert isinstance(idx, pd.DatetimeIndex)  # minute timestamps, not collapsed to a day
    assert list(idx.minute) == [30, 31, 32]  # three distinct minutes -> three rows
    assert float(panel["AAPL"].iloc[-1]) == 182.5
