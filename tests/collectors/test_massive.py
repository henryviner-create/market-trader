"""The Massive price client: grouped-daily, universe filtering, weekend skip, rate limit."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from market_trader.collectors.massive import MassiveClient, MassiveError
from market_trader.collectors.prices import PriceCollector
from market_trader.core.synthetic import PRICE_DATASET


def test_requires_api_key() -> None:
    with pytest.raises(MassiveError):
        MassiveClient("")


def test_grouped_daily_filters_universe_and_builds_bars() -> None:
    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        assert headers["x-api-key"] == "k"  # auth header carried
        assert "2024-06-03" in url  # the requested trading day
        return 200, {
            "status": "OK",
            "results": [
                {"T": "ABC", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 1000},
                {"T": "ZZZ", "c": 9.0},  # outside the universe -> filtered out
            ],
        }

    client = MassiveClient("k", transport=transport, min_interval_seconds=0.0)
    bars = client.fetch_daily_bars(["ABC"], start=date(2024, 6, 3), end=date(2024, 6, 3))  # Monday
    assert len(bars) == 1
    assert bars[0].symbol == "ABC" and bars[0].close == 1.5 and bars[0].volume == 1000

    # the bars normalize into the same price dataset the backtester reads
    obs = PriceCollector().normalize(bars)
    assert obs[0].dataset == PRICE_DATASET and obs[0].entity_id == "ABC"


def test_skips_weekends_with_no_calls() -> None:
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return 200, {"results": []}

    client = MassiveClient("k", transport=transport, min_interval_seconds=0.0)
    client.fetch_daily_bars(["ABC"], start=date(2024, 6, 8), end=date(2024, 6, 9))  # Sat, Sun
    assert calls == []  # no session -> no API spend


def test_rate_limit_delay_is_applied_between_trading_days() -> None:
    waits: list[float] = []

    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        return 200, {"results": []}

    client = MassiveClient("k", transport=transport, min_interval_seconds=12.0, sleep=waits.append)
    client.fetch_daily_bars(["ABC"], start=date(2024, 6, 3), end=date(2024, 6, 5))  # Mon-Wed
    # 3 trading days, 3 calls; the first needs no wait, the next two are throttled ~12s apart
    assert len(waits) == 2 and all(w > 0 for w in waits)


def test_non_200_day_is_skipped_not_fatal() -> None:
    def transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        return 403, {"error": "forbidden"}

    client = MassiveClient("k", transport=transport, min_interval_seconds=0.0)
    bars = client.fetch_daily_bars(["ABC"], start=date(2024, 6, 3), end=date(2024, 6, 3))
    assert bars == []  # a bad day is logged and skipped, never aborts the backfill
