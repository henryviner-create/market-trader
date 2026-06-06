"""Local-file daily-price ingestion — the offline path when a source IP-blocks the box.

Stooq has deep, survivorship-correct daily history (it retains delisted names), which makes
it the ideal cold-start backfill for the ``price.ohlcv`` dataset the backtester/replay read.
But Stooq serves a bot-challenge page to **datacenter IPs**, so a direct sweep from a cloud
droplet returns zero bars (see :mod:`market_trader.collectors.stooq`). The fix is to fetch the
CSVs on a machine that *can* reach the source — a browser or a laptop on a residential IP —
and ingest the local files here. The bars flow through the same
:meth:`PriceCollector.normalize <market_trader.collectors.prices.PriceCollector.normalize>`,
so a local CSV shares identical point-in-time machinery with every other source.

Two on-disk layouts are understood, auto-detected from the header line:

* **Stooq per-symbol download** (``/q/d/l/`` CSV): ``Date,Open,High,Low,Close,Volume``. The
  symbol is *not* in the file, so it comes from the filename (``aapl.us.csv`` -> ``AAPL``) or
  an explicit ``symbol`` argument. Reuses Stooq's own row parser so the two paths can never
  drift.
* **Stooq bulk DB** (``/db/h/`` ``.txt``): ``<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,...,<CLOSE>,
  <VOL>,<OPENINT>`` with the ticker *in each row* and ``YYYYMMDD`` dates — so one file can
  carry many symbols (the whole delisted-inclusive universe), and the filename is ignored.

As with every collector, junk/empty/``No data`` bodies yield zero bars rather than raising,
and ``close`` is the only required field (a fabricated bar is worse than a missing one).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from market_trader.collectors.prices import PriceBar

# Reuse Stooq's tested d/l-row parser and numeric cell parser so the per-symbol path
# behaves identically to the live Stooq sweep (no second, drifting copy of the logic).
from market_trader.collectors.stooq import _opt_float, _parse_csv
from market_trader.observability import get_logger

_log = get_logger("csv_prices")

_PRICE_SUFFIXES = (".csv", ".txt")
# Trailing tokens a Stooq per-symbol filename may carry beyond the bare ticker.
_TICKER_TRAILERS = ("_us_d", "_us", "_d")


def symbol_from_filename(name: str) -> str:
    """Best-effort ticker from a per-symbol filename (``aapl.us.csv`` -> ``AAPL``).

    Drops the extension and a Stooq ``.us`` / ``_us_d`` style suffix. A ticker with an
    internal dot (e.g. ``BRK.B``) can't be recovered from the filename — pass ``symbol``
    explicitly for those, or use the bulk format, which carries the ticker per row.
    """
    stem = Path(name).stem.split(".")[0]  # "aapl.us" -> "aapl"; "MSFT" -> "MSFT"
    low = stem.lower()
    for suf in _TICKER_TRAILERS:
        if low.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.upper()


def _parse_bulk_csv(body: str) -> list[PriceBar]:
    """Parse the Stooq bulk-DB layout (ticker in each row, ``YYYYMMDD`` dates)."""
    out: list[PriceBar] = []
    for row in csv.DictReader(body.splitlines()):
        ticker = (row.get("<TICKER>") or "").strip()
        raw_date = (row.get("<DATE>") or "").strip()
        close = _opt_float(row.get("<CLOSE>"))
        if not ticker or not raw_date or close is None:
            continue
        try:
            bar_date = datetime.strptime(raw_date, "%Y%m%d").date()
        except ValueError:
            continue
        out.append(
            PriceBar(
                date=bar_date,
                symbol=ticker.split(".")[0].upper(),  # "AAPL.US" -> "AAPL"
                close=close,
                open=_opt_float(row.get("<OPEN>")),
                high=_opt_float(row.get("<HIGH>")),
                low=_opt_float(row.get("<LOW>")),
                volume=_opt_float(row.get("<VOL>")),
            )
        )
    return out


def parse_price_csv(body: str, *, default_symbol: str | None = None) -> list[PriceBar]:
    """Parse a daily-OHLCV CSV body into :class:`PriceBar`s, auto-detecting the layout.

    Bulk-DB rows carry their own ticker; per-symbol (Stooq d/l) rows do not, so a
    ``default_symbol`` is required for that layout (an empty one yields zero bars rather
    than mislabelling). An empty / header-only / ``No data`` body returns ``[]``.
    """
    if not body or not body.strip():
        return []
    header = body.splitlines()[0]
    if "<TICKER>" in header or "<DATE>" in header:
        return _parse_bulk_csv(body)
    if not default_symbol:
        return []  # per-symbol layout with no symbol to assign -> skip, don't guess
    return _parse_csv(body, symbol=default_symbol.upper())


def read_price_csv(path: Path, *, symbol: str | None = None) -> list[PriceBar]:
    """Read one CSV file into bars; for the per-symbol layout, infer the symbol from the name."""
    body = path.read_text(encoding="utf-8", errors="ignore")
    return parse_price_csv(body, default_symbol=symbol or symbol_from_filename(path.name))


def iter_price_files(path: Path) -> list[Path]:
    """A single CSV/TXT file, or every CSV/TXT directly under a directory (sorted)."""
    if path.is_dir():
        return sorted(
            f for f in path.iterdir() if f.is_file() and f.suffix.lower() in _PRICE_SUFFIXES
        )
    return [path]


def read_price_files(paths: Iterable[Path], *, symbol: str | None = None) -> list[PriceBar]:
    """Read many files into one bar list. ``symbol`` applies only to a single per-symbol file."""
    files = list(paths)
    one = len(files) == 1
    bars: list[PriceBar] = []
    for f in files:
        try:
            bars.extend(read_price_csv(f, symbol=symbol if one else None))
        except Exception:  # best-effort: a single unreadable file never aborts the batch
            _log.warning("csv_read_error", path=str(f))
    return bars
