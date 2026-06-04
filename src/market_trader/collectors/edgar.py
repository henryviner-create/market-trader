"""SEC EDGAR insider transactions (Form 4).

Insider Form 4s carry a ~2-day disclosure lag — far more tradable than the
~45-day congressional lag — so they are a higher-value flow signal. ``event_time``
is the transaction date; ``knowledge_time`` is the filing acceptance date. A
purchase (code ``P``) is the informative event; cluster-buys are flagged later.

``normalize`` works on parsed Form 4 records. Live ``fetch`` (EDGAR full-text
search / submissions API + XML parsing) is wired when scheduling collection.
"""

from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close
from market_trader.observability import get_logger

FORM4_DATASET = "filing.form4"


class Form4Record(BaseModel):
    issuer_ticker: str
    insider_name: str
    transaction_code: str  # P=purchase, S=sale, A=grant, M=option exercise, ...
    transaction_date: date
    filing_date: date  # acceptance date (~2 days after the transaction)
    insider_title: str | None = None
    shares: float | None = None
    price_per_share: float | None = None


class Form4Collector(Collector):
    source = "edgar"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        records = [r if isinstance(r, Form4Record) else Form4Record.model_validate(r) for r in raw]
        out: list[Observation] = []
        for r in records:
            out.append(
                Observation(
                    source=self.source,
                    dataset=FORM4_DATASET,
                    entity_type="equity",
                    entity_id=r.issuer_ticker.upper(),
                    ref=f"{r.insider_name}|{r.transaction_code}",
                    event_time=day_close(r.transaction_date),
                    knowledge_time=day_close(r.filing_date),
                    value={
                        "transaction_code": r.transaction_code,
                        "is_purchase": r.transaction_code.upper() == "P",
                        "shares": r.shares,
                        "price_per_share": r.price_per_share,
                        "insider_name": r.insider_name,
                        "insider_title": r.insider_title,
                    },
                    metadata={
                        "filing_lag_days": (r.filing_date - r.transaction_date).days,
                        "parser_version": self.parser_version,
                    },
                )
            )
        return out


# --- live fetch (SEC EDGAR) -------------------------------------------------

_log = get_logger("edgar")

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
_SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# (url) -> response body text (JSON or XML). Injectable so parsing is offline-testable.
EdgarTransport = Callable[[str], str]


def _edgar_get(url: str, *, user_agent: str, timeout: float) -> str:
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


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class EdgarClient:
    """Fetch recent SEC Form-4 insider filings for a set of tickers.

    Mirrors :class:`~market_trader.collectors.gdelt.GdeltClient`: stdlib ``urllib``,
    an injectable ``transport`` (so the XML/JSON parsing is offline-testable), a
    per-request timeout, and a wall-clock ``budget_seconds`` so a slow SEC endpoint
    can never stall a trading cycle. SEC requires a descriptive ``User-Agent``.

    v1 reads each issuer's ``submissions`` feed (the most recent filings) and parses
    the non-derivative table of each Form 4. Deeper historical backfill (the older
    ``filings.files`` shards) is a later enhancement.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        transport: EdgarTransport | None = None,
        timeout_seconds: float = 15.0,
        budget_seconds: float = 90.0,
        max_filings_per_symbol: int = 250,
    ) -> None:
        self._get = transport or (
            lambda url: _edgar_get(url, user_agent=user_agent, timeout=timeout_seconds)
        )
        self._budget = budget_seconds
        self._max_filings = max_filings_per_symbol
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
                _log.warning("edgar_cik_map_failed", error=str(exc))
                self._cik = {}
        return self._cik

    def ticker_universe(self) -> list[str]:
        """All SEC-filer tickers — the candidate set for a liquidity screen.

        These are exactly the names that can *have* Form-4 filings, so they are the
        natural starting population for an insider-signal universe.
        """
        return sorted(self._cik_map().keys())

    def fetch_for_symbols(
        self, symbols: Sequence[str], *, lookback_days: int = 400
    ) -> list[Form4Record]:
        """Best-effort per-symbol Form-4 fetch, bounded by a wall-clock budget."""
        cik_map = self._cik_map()
        cutoff = date.today() - timedelta(days=lookback_days)
        out: list[Form4Record] = []
        start = time.monotonic()
        fetched = 0
        for sym in symbols:
            if time.monotonic() - start > self._budget:
                _log.warning("edgar_budget_exceeded", fetched=fetched, total=len(symbols))
                break
            cik = cik_map.get(str(sym).upper())
            if not cik:
                continue
            try:
                out.extend(self._fetch_symbol(str(sym), cik, cutoff))
                fetched += 1
            except Exception:  # one bad symbol never aborts the batch
                continue
        _log.info(
            "edgar_fetch",
            symbols=fetched,
            records=len(out),
            elapsed_s=round(time.monotonic() - start, 1),
        )
        return out

    def _fetch_symbol(self, ticker: str, cik: str, cutoff: date) -> list[Form4Record]:
        subs = json.loads(self._get(_SEC_SUBMISSIONS_URL.format(cik=cik)))
        filings = subs.get("filings", {})
        out: list[Form4Record] = []
        seen = self._process_block(filings.get("recent", {}), cik, ticker, cutoff, out, 0)
        # Older filings live in additional shards (`filings.files`) — pull those whose
        # date range reaches into the lookback. Without this we only ever see ~the last
        # year for active filers: too few independent decision dates to confirm an edge.
        for shard in filings.get("files", []):
            if seen >= self._max_filings:
                break
            to = _parse_date(shard.get("filingTo"))
            name = shard.get("name")
            if to is None or to < cutoff or not name:
                continue
            try:
                block = json.loads(self._get(_SEC_SUBMISSIONS_FILE_URL.format(name=name)))
                seen = self._process_block(block, cik, ticker, cutoff, out, seen)
            except Exception:  # skip an unreadable shard
                continue
        return out

    def _process_block(
        self,
        block: dict[str, Any],
        cik: str,
        ticker: str,
        cutoff: date,
        out: list[Form4Record],
        seen: int,
    ) -> int:
        """Parse the Form-4 filings in one parallel-array submissions block."""
        rows = zip(
            block.get("form", []),
            block.get("accessionNumber", []),
            block.get("filingDate", []),
            block.get("primaryDocument", []),
            strict=False,
        )
        for form, acc, fdate, doc in rows:
            if form != "4":
                continue
            fd = _parse_date(fdate)
            if fd is None or fd < cutoff:
                continue
            if seen >= self._max_filings:
                break
            seen += 1
            try:
                xml = self._get(self._doc_url(cik, str(acc), str(doc)))
                out.extend(self.parse_form4_xml(xml, filing_date=fd, fallback_ticker=ticker))
            except Exception:  # skip an unparseable filing
                continue
        return seen

    @staticmethod
    def _doc_url(cik: str, accession: str, primary_document: str) -> str:
        # primaryDocument is often the XSL-rendered path (e.g. "xslF345X05/foo.xml");
        # the raw ownership XML is the same name without the directory prefix.
        name = primary_document.split("/")[-1] if primary_document else ""
        return _SEC_ARCHIVE_URL.format(cik=int(cik), acc=accession.replace("-", ""), doc=name)

    @staticmethod
    def parse_form4_xml(
        xml_text: str, *, filing_date: date, fallback_ticker: str | None = None
    ) -> list[Form4Record]:
        """Parse a Form-4 ownership XML into one record per non-derivative transaction."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        def _text(node: Any, path: str) -> str | None:
            el = node.find(path)
            return el.text.strip() if el is not None and el.text else None

        ticker = _text(root, "issuer/issuerTradingSymbol") or (fallback_ticker or "")
        if not ticker:
            return []
        owner = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName") or "unknown"
        title = _text(root, "reportingOwner/reportingOwnerRelationship/officerTitle")
        out: list[Form4Record] = []
        for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
            code = _text(txn, "transactionCoding/transactionCode")
            tdate = _parse_date(_text(txn, "transactionDate/value"))
            if not code or tdate is None:
                continue
            out.append(
                Form4Record(
                    issuer_ticker=ticker.upper(),
                    insider_name=owner,
                    transaction_code=code,
                    transaction_date=tdate,
                    filing_date=filing_date,
                    insider_title=title,
                    shares=_to_float(_text(txn, "transactionAmounts/transactionShares/value")),
                    price_per_share=_to_float(
                        _text(txn, "transactionAmounts/transactionPricePerShare/value")
                    ),
                )
            )
        return out
