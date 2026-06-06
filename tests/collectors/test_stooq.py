"""Stooq client: CSV parsing + PriceBar-compatible records, fully offline.

The client takes an injectable CSV transport, so we exercise the URL it builds and
the :class:`PriceBar`s it parses with zero network — then prove those bars flow
cleanly through the same ``PriceCollector`` normalisation that synthetic / Alpaca /
yfinance bars use. An injected no-op ``sleep`` keeps the rate-limited sweep instant.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from market_trader.collectors import PriceCollector
from market_trader.collectors.prices import PriceBar
from market_trader.collectors.stooq import StooqClient
from market_trader.core.synthetic import PRICE_DATASET

_AAPL_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2023-06-01,180.0,182.0,179.0,181.0,1000\n"
    "2023-06-02,181.0,184.0,180.5,183.5,1200\n"
)
_MSFT_CSV = "Date,Open,High,Low,Close,Volume\n2023-06-01,330.0,333.0,329.0,332.0,800\n"
# Stooq returns this body verbatim for an unknown/invalid ticker.
_NO_DATA_CSV = "No data\n"
_HEADER_ONLY_CSV = "Date,Open,High,Low,Close,Volume\n"

_BODIES = {"aapl": _AAPL_CSV, "msft": _MSFT_CSV}


def _transport(bodies: dict[str, str] | None = None) -> tuple[Callable[[str], str], list[str]]:
    """Stub transport: route on the lowercased ticker in the URL, recording calls."""
    table = _BODIES if bodies is None else bodies
    calls: list[str] = []

    def transport(url: str) -> str:
        calls.append(url)
        for ticker, body in table.items():
            if f"s={ticker}.us" in url:
                return body
        return _NO_DATA_CSV

    return transport, calls


def _no_sleep() -> tuple[Callable[[float], None], list[float]]:
    slept: list[float] = []
    return (lambda secs: slept.append(secs)), slept


def test_parses_normal_csv_into_price_bars() -> None:
    transport, calls = _transport()
    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)

    bars = client.fetch_daily_bars(["AAPL"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert len(bars) == 2
    assert all(isinstance(b, PriceBar) for b in bars)
    first = bars[0]
    assert first.date == date(2023, 6, 1)
    assert first.symbol == "AAPL"
    assert first.close == 181.0 and first.volume == 1000.0
    assert first.open == 180.0 and first.high == 182.0 and first.low == 179.0
    # lowercased ticker + .us suffix + dash-free date window
    assert "s=aapl.us" in calls[0]
    assert "d1=20230601" in calls[0] and "d2=20230602" in calls[0]


def test_unknown_symbol_yields_no_bars() -> None:
    transport, _ = _transport({})  # everything routes to the "No data" body
    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)

    bars = client.fetch_daily_bars(["NOPE"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert bars == []  # missing symbol is normal, not a crash


def test_header_only_body_yields_no_bars() -> None:
    transport, _ = _transport({"zzzz": _HEADER_ONLY_CSV})
    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)

    bars = client.fetch_daily_bars(["ZZZZ"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert bars == []  # header with no rows parses cleanly to zero bars


def test_multiple_symbols_one_request_each() -> None:
    transport, calls = _transport()
    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)

    bars = client.fetch_daily_bars(["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert len(calls) == 2  # per-symbol: one HTTP request each, no bulk endpoint
    assert len(bars) == 3  # 2 AAPL + 1 MSFT
    assert {b.symbol for b in bars} == {"AAPL", "MSFT"}


def test_one_symbol_erroring_does_not_abort_batch() -> None:
    calls: list[str] = []

    def transport(url: str) -> str:
        calls.append(url)
        if "s=bad.us" in url:
            raise RuntimeError("boom")
        return _AAPL_CSV

    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)

    bars = client.fetch_daily_bars(["BAD", "AAPL"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert len(calls) == 2  # tried both
    assert len(bars) == 2 and all(b.symbol == "AAPL" for b in bars)  # BAD skipped, AAPL kept


def test_rate_limit_sleep_runs_between_symbols() -> None:
    transport, _ = _transport()
    sleep, slept = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep, request_delay_seconds=0.25)

    client.fetch_daily_bars(["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    assert slept == [0.25]  # delay paid once between two symbols, not after the last


def test_bars_normalize_through_price_collector() -> None:
    transport, _ = _transport()
    sleep, _ = _no_sleep()
    client = StooqClient(transport=transport, sleep=sleep)
    bars = client.fetch_daily_bars(["AAPL", "MSFT"], start=date(2023, 6, 1), end=date(2023, 6, 2))

    observations = PriceCollector().normalize(bars)  # same path as synthetic/yfinance

    assert observations
    assert all(o.dataset == PRICE_DATASET for o in observations)
    aapl = [o for o in observations if o.entity_id == "AAPL"]
    assert len(aapl) == 2
    assert all(o.knowledge_time == o.event_time for o in aapl)  # known at the close
    assert aapl[0].value["close"] == 181.0 and aapl[0].value["volume"] == 1000.0
