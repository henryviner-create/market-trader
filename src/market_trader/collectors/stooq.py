"""Stooq daily-price client (CSV) — stdlib urllib, offline-testable.

Stooq is a free, no-key source with surprisingly deep daily history, which makes
it a good cold-start backfill for the same ``price.ohlcv`` dataset the backtester
reads: the :class:`~market_trader.collectors.prices.PriceBar` objects produced here
flow through :meth:`PriceCollector.normalize <market_trader.collectors.prices.PriceCollector.normalize>`,
so Stooq bars share identical point-in-time machinery with synthetic / Alpaca /
yfinance bars and need no special-casing downstream.

Two things shape this module. First, Stooq speaks **CSV text**, not JSON, so the
injectable transport is ``Callable[[str], str]`` (url -> CSV body) rather than the
JSON transport the other clients use. Second, Stooq is strictly **per-symbol**
(no bulk endpoint), so a broad-universe sweep is N HTTP requests; to stay a polite
citizen we space requests with a small inter-request ``sleep`` and bound the whole
sweep with a wall-clock ``budget``. As with the other best-effort collectors, one
symbol erroring (or returning an empty/``No data`` body) must never abort the batch.
"""

from __future__ import annotations

import csv
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, datetime

from market_trader.collectors.prices import PriceBar
from market_trader.observability import get_logger

STOOQ_CSV_URL = "https://stooq.com/q/d/l/"

_log = get_logger("stooq")

# (url) -> CSV body text. Injectable so the sweep is fully offline-testable.
CsvTransport = Callable[[str], str]


class StooqError(RuntimeError):
    pass


def _stooq_get(url: str, *, timeout: float = 10.0) -> str:
    """Real transport: fetch ``url`` and return the decoded CSV body.

    A descriptive User-Agent is sent because Stooq serves a bot-challenge HTML page
    to clients that look like default scripts, which would parse as zero rows.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "market-trader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # fixed https host
        raw = resp.read()
    return raw.decode("utf-8", "ignore") if raw else ""


def _csv_url(ticker: str, *, start: date, end: date) -> str:
    """Stooq daily-CSV endpoint for one symbol.

    Stooq US tickers are lowercase with a ``.us`` suffix, and the date window is
    ``YYYYMMDD`` with no separators (``d1``=from, ``d2``=to).
    """
    d1 = start.strftime("%Y%m%d")
    d2 = end.strftime("%Y%m%d")
    return f"{STOOQ_CSV_URL}?s={ticker.lower()}.us&i=d&d1={d1}&d2={d2}"


def _opt_float(raw: str | None) -> float | None:
    """Parse an optional OHLCV cell; Stooq writes ``N/D`` (or blank) for gaps."""
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.upper() == "N/D":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_csv(body: str, *, symbol: str) -> list[PriceBar]:
    """Turn one symbol's CSV body into :class:`PriceBar`s, skipping junk rows.

    A missing/invalid symbol comes back as ``No data`` or a header-only body, so an
    empty result is normal — not an error — and yields zero bars. ``close`` is the
    only required field; a row without a parseable date or close is dropped rather
    than guessed, because a fabricated bar is worse than a missing one.
    """
    out: list[PriceBar] = []
    for row in csv.DictReader(body.splitlines()):
        raw_date = (row.get("Date") or "").strip()
        close = _opt_float(row.get("Close"))
        if not raw_date or close is None:
            continue
        try:
            bar_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        out.append(
            PriceBar(
                date=bar_date,
                symbol=symbol,
                close=close,
                open=_opt_float(row.get("Open")),
                high=_opt_float(row.get("High")),
                low=_opt_float(row.get("Low")),
                volume=_opt_float(row.get("Volume")),
            )
        )
    return out


class StooqClient:
    """Fetch deep daily history from Stooq's free CSV endpoint (no key required).

    The injected ``transport`` (url -> CSV body) keeps the sweep offline-testable;
    the injected ``sleep`` makes the inter-request pacing instant under test. The
    ``budget`` is load-bearing: Stooq is per-symbol, so a sweep over a ~140-name
    universe is ~140 sequential requests, and the free endpoint throttles — once
    the wall-clock budget elapses the sweep stops and the remaining names are
    skipped this cycle rather than stalling a whole collection run.
    """

    def __init__(
        self,
        *,
        base_url: str = STOOQ_CSV_URL,
        transport: CsvTransport | None = None,
        timeout_seconds: float = 10.0,
        request_delay_seconds: float = 0.5,
        budget_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base = base_url
        self._get = transport or (lambda url: _stooq_get(url, timeout=timeout_seconds))
        self._delay = request_delay_seconds
        self._budget = budget_seconds
        self._sleep = sleep

    def fetch_symbol(self, symbol: str, *, start: date, end: date) -> list[PriceBar]:
        """Daily bars for a single symbol over ``[start, end]`` (CSV parsed offline-safe)."""
        body = self._get(_csv_url(symbol, start=start, end=end))
        return _parse_csv(body, symbol=symbol)

    def fetch_daily_bars(self, symbols: Sequence[str], *, start: date, end: date) -> list[PriceBar]:
        """Best-effort per-symbol sweep, bounded by a wall-clock budget.

        One request per symbol (Stooq has no bulk endpoint). One symbol erroring or
        returning ``No data`` never aborts the batch — it is skipped and the sweep
        continues. Between symbols we ``sleep`` a small delay to stay polite, but
        skip the delay after the final symbol so we never pay it for nothing.
        """
        out: list[PriceBar] = []
        start_clock = time.monotonic()
        fetched = 0
        total = len(symbols)
        for i, sym in enumerate(symbols):
            if time.monotonic() - start_clock > self._budget:
                _log.warning(
                    "stooq_budget_exceeded",
                    fetched=fetched,
                    total=total,
                    budget_seconds=self._budget,
                )
                break
            try:
                out.extend(self.fetch_symbol(sym, start=start, end=end))
                fetched += 1
            except Exception:  # best-effort batch: skip a symbol that errors
                continue
            if self._delay > 0 and i < total - 1:
                self._sleep(self._delay)
        _log.info(
            "stooq_fetch",
            symbols=fetched,
            bars=len(out),
            elapsed_s=round(time.monotonic() - start_clock, 1),
        )
        return out
