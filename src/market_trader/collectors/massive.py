"""Massive market-data collector (Polygon-compatible REST; free tier = EOD, 5 calls/min).

Massive's REST API mirrors Polygon's shape (``{status, results, request_id}``; the
grouped-daily endpoint returns the whole market's OHLC for one date in a *single* call).
The free tier is EOD US equities at 5 calls/min with ~2y history — exactly enough for the
daily book's **price coverage** (cleaner small-cap coverage than the free IEX feed, which
silently drops thin names). So we use grouped-daily (one call per *trading day*, never per
symbol) with a hard inter-call delay to stay inside the rate limit, and reuse
:class:`PriceCollector` so Massive prices flow through the identical point-in-time
machinery as every other source.

Auth is the ``x-api-key`` header. ``base_url`` / ``grouped_path`` are configurable because
this is built to the documented Polygon-compatible shape and the exact paths should be
confirmed against the account; the transport is injectable so the logic is fully tested
offline (no network, no key).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

from market_trader.collectors.prices import PriceBar
from market_trader.observability import get_logger

MASSIVE_BASE_URL = "https://api.massive.com"
# Polygon-style grouped daily aggregates: every ticker's OHLC for one date, one request.
GROUPED_DAILY_PATH = "/v2/aggs/grouped/locale/us/market/stocks/{day}"

_log = get_logger("massive")

# (url, headers) -> (status, parsed-json)
MassiveTransport = Callable[[str, dict[str, str]], tuple[int, dict[str, Any]]]


class MassiveError(RuntimeError):
    pass


def _urllib_get(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30.0) as resp:  # fixed https host
        raw = resp.read()
    return resp.status, (json.loads(raw) if raw else {})


class MassiveClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = MASSIVE_BASE_URL,
        grouped_path: str = GROUPED_DAILY_PATH,
        transport: MassiveTransport | None = None,
        min_interval_seconds: float = 12.0,  # 5 calls/min free tier -> >=12s apart
        budget_seconds: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise MassiveError("Massive API key required (MT_MASSIVE_API_KEY)")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._grouped = grouped_path
        self._get = transport or _urllib_get
        self._min_interval = min_interval_seconds
        self._budget = budget_seconds
        self._sleep = sleep

    def _grouped_daily(self, day: date) -> list[dict[str, Any]]:
        url = f"{self._base}{self._grouped.format(day=day.isoformat())}"
        status, payload = self._get(url, {"x-api-key": self._key})
        if status != 200:
            raise MassiveError(f"grouped-daily {day.isoformat()} -> HTTP {status}")
        results = payload.get("results")
        return results if isinstance(results, list) else []

    def fetch_daily_bars(self, symbols: Sequence[str], *, start: date, end: date) -> list[PriceBar]:
        """One grouped-daily call per trading day over ``[start, end]``, filtered to ``symbols``.

        Rate-limited to the free-tier 5 calls/min and bounded by a wall-clock budget;
        weekends are skipped (no session). A day that errors is logged and skipped — the
        backfill is resumable on a later run.
        """
        wanted = {s.upper() for s in symbols}
        out: list[PriceBar] = []
        started = time.monotonic()
        last_call = 0.0
        day = start
        while day <= end:
            if time.monotonic() - started > self._budget:
                _log.warning("massive_budget_exceeded", through=day.isoformat(), bars=len(out))
                break
            if day.weekday() < 5:  # skip weekends (no session)
                wait = self._min_interval - (time.monotonic() - last_call)
                if wait > 0:
                    self._sleep(wait)  # respect the rate limit
                last_call = time.monotonic()
                try:
                    for r in self._grouped_daily(day):
                        ticker = str(r.get("T", "")).upper()
                        if ticker in wanted and r.get("c") is not None:
                            out.append(
                                PriceBar(
                                    date=day,
                                    symbol=ticker,
                                    close=float(r["c"]),
                                    open=r.get("o"),
                                    high=r.get("h"),
                                    low=r.get("l"),
                                    volume=r.get("v"),
                                )
                            )
                except MassiveError as exc:
                    _log.warning("massive_day_failed", day=day.isoformat(), error=str(exc))
            day += timedelta(days=1)
        _log.info("massive_fetch", bars=len(out), symbols=len(wanted))
        return out
