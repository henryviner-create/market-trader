"""SEC XBRL company fundamentals (the ``companyfacts`` API).

One request per company returns its entire reported XBRL history. We extract the
*quarterly* diluted-EPS series — which powers both a **value** signal (trailing-12-month
earnings yield) and a **PEAD** signal (standardized earnings surprise). The crucial
field is ``filed``: the date a figure became public, used as ``knowledge_time`` so the
features can never see an earnings number before it was announced.

Mirrors :class:`~market_trader.collectors.edgar.EdgarClient`: stdlib ``urllib``, an
injectable ``transport`` (offline-testable parsing), a per-request timeout, and a
wall-clock budget. SEC requires a descriptive ``User-Agent``; no account/key needed.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close
from market_trader.observability import get_logger

FUNDAMENTAL_DATASET = "fundamental.quarterly"

_log = get_logger("fundamentals")

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# (url) -> response body text (JSON). Injectable so parsing is offline-testable.
FundamentalsTransport = Callable[[str], str]


def _sec_get(url: str, *, user_agent: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})  # SEC requires a UA
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class FundamentalRecord(BaseModel):
    ticker: str
    period_end: date  # the fiscal quarter end (event_time)
    filed_date: date  # when it was first disclosed (~knowledge_time)
    eps: float  # diluted (or basic) EPS for the quarter
    fiscal_period: str | None = None  # Q1/Q2/Q3/Q4


class FundamentalsCollector(Collector):
    source = "sec_xbrl"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        records = [
            r if isinstance(r, FundamentalRecord) else FundamentalRecord.model_validate(r)
            for r in raw
        ]
        out: list[Observation] = []
        for r in records:
            out.append(
                Observation(
                    source=self.source,
                    dataset=FUNDAMENTAL_DATASET,
                    entity_type="equity",
                    entity_id=r.ticker.upper(),
                    ref=r.period_end.isoformat(),  # one logical row per fiscal quarter
                    event_time=day_close(r.period_end),
                    knowledge_time=day_close(r.filed_date),  # not knowable before it was filed
                    value={"eps": r.eps, "fiscal_period": r.fiscal_period},
                    metadata={"parser_version": self.parser_version},
                )
            )
        return out


class FundamentalsClient:
    """Fetch quarterly fundamentals (diluted EPS) for a set of tickers from SEC XBRL."""

    def __init__(
        self,
        *,
        user_agent: str,
        transport: FundamentalsTransport | None = None,
        timeout_seconds: float = 15.0,
        budget_seconds: float = 120.0,
    ) -> None:
        self._get = transport or (
            lambda url: _sec_get(url, user_agent=user_agent, timeout=timeout_seconds)
        )
        self._budget = budget_seconds
        self._cik: dict[str, str] | None = None

    def _cik_map(self) -> dict[str, str]:
        if self._cik is None:
            try:
                data = json.loads(self._get(_SEC_TICKERS_URL))
                self._cik = {
                    str(row["ticker"]).upper(): f"{int(row['cik_str']):010d}"
                    for row in data.values()
                }
            except Exception as exc:  # no CIK map -> nothing to fetch (degrade gracefully)
                _log.warning("fundamentals_cik_map_failed", error=str(exc))
                self._cik = {}
        return self._cik

    def fetch_for_symbols(self, symbols: Sequence[str]) -> list[FundamentalRecord]:
        """Best-effort per-symbol fundamentals fetch, bounded by a wall-clock budget.

        One ``companyfacts`` request per company (cheap), so a universe backfills fast.
        """
        cik_map = self._cik_map()
        out: list[FundamentalRecord] = []
        start = time.monotonic()
        fetched = 0
        for sym in symbols:
            if time.monotonic() - start > self._budget:
                _log.warning("fundamentals_budget_exceeded", fetched=fetched, total=len(symbols))
                break
            cik = cik_map.get(str(sym).upper())
            if not cik:
                continue
            try:
                out.extend(self._fetch_symbol(str(sym), cik))
                fetched += 1
            except Exception:  # one bad symbol never aborts the batch
                continue
        _log.info(
            "fundamentals_fetch",
            symbols=fetched,
            records=len(out),
            elapsed_s=round(time.monotonic() - start, 1),
        )
        return out

    def _fetch_symbol(self, ticker: str, cik: str) -> list[FundamentalRecord]:
        data = json.loads(self._get(_COMPANYFACTS_URL.format(cik=cik)))
        return self.parse_companyfacts(data, fallback_ticker=ticker)

    @staticmethod
    def parse_companyfacts(
        data: dict[str, Any], *, fallback_ticker: str | None = None
    ) -> list[FundamentalRecord]:
        """Extract the quarterly diluted-EPS series from a ``companyfacts`` payload.

        Keeps only ~3-month-period facts (true quarterly, not year-to-date or annual),
        and for each fiscal quarter keeps the *earliest* filing — the original earnings
        announcement — so ``knowledge_time`` is when the surprise actually hit the tape.
        """
        ticker = (fallback_ticker or "").upper()
        if not ticker:
            return []
        gaap = ((data.get("facts") or {}).get("us-gaap")) or {}
        concept = gaap.get("EarningsPerShareDiluted") or gaap.get("EarningsPerShareBasic") or {}
        entries = (concept.get("units") or {}).get("USD/shares") or []

        by_quarter: dict[str, dict[str, Any]] = {}
        for e in entries:
            start, end, filed, val = e.get("start"), e.get("end"), e.get("filed"), e.get("val")
            sd, ed, fd = _parse_date(start), _parse_date(end), _parse_date(filed)
            if ed is None or fd is None or sd is None or val is None:
                continue
            if not (80 <= (ed - sd).days <= 100):  # quarterly only (~3 months)
                continue
            key = ed.isoformat()
            prior = by_quarter.get(key)
            if prior is None or fd < prior["filed"]:  # the original announcement
                by_quarter[key] = {"end": ed, "filed": fd, "eps": float(val), "fp": e.get("fp")}
        return [
            FundamentalRecord(
                ticker=ticker,
                period_end=r["end"],
                filed_date=r["filed"],
                eps=r["eps"],
                fiscal_period=(str(r["fp"]) if r["fp"] else None),
            )
            for r in by_quarter.values()
        ]
