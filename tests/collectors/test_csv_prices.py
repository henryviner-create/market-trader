"""Local-CSV price ingestion: Stooq per-symbol d/l + bulk DB layouts, fully offline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from market_trader.collectors.csv_prices import (
    iter_price_files,
    parse_price_csv,
    read_price_csv,
    read_price_files,
    symbol_from_filename,
)

# Stooq per-symbol download (/q/d/l/): the symbol is NOT in the file.
DL_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,185.0,187.0,184.0,186.0,1000000
2024-01-03,186.0,188.0,185.5,187.5,1100000
"""

# Stooq bulk DB (/db/h/): ticker in each row, YYYYMMDD dates, one file => many symbols.
BULK_TXT = """<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20240102,000000,185.0,187.0,184.0,186.0,1000000,0
MSFT.US,D,20240102,000000,370.0,373.0,369.0,372.0,900000,0
AAPL.US,D,20240103,000000,186.0,188.0,185.5,187.5,1100000,0
"""


def test_symbol_from_filename() -> None:
    assert symbol_from_filename("aapl.us.csv") == "AAPL"
    assert symbol_from_filename("MSFT.csv") == "MSFT"
    assert symbol_from_filename("nvda_us_d.csv") == "NVDA"


def test_parse_dl_layout_uses_default_symbol() -> None:
    bars = parse_price_csv(DL_CSV, default_symbol="AAPL")
    assert [b.symbol for b in bars] == ["AAPL", "AAPL"]
    assert bars[0].date == date(2024, 1, 2)
    assert bars[0].close == 186.0 and bars[0].open == 185.0
    assert bars[1].close == 187.5


def test_parse_dl_layout_without_symbol_yields_nothing() -> None:
    # Per-symbol layout with no symbol to assign: skip rather than mislabel.
    assert parse_price_csv(DL_CSV) == []


def test_parse_bulk_layout_carries_per_row_ticker() -> None:
    bars = parse_price_csv(BULK_TXT)
    by_sym: dict[str, list] = {}
    for b in bars:
        by_sym.setdefault(b.symbol, []).append(b)
    assert set(by_sym) == {"AAPL", "MSFT"}  # ".US" stripped, one file -> two symbols
    assert len(by_sym["AAPL"]) == 2
    assert by_sym["AAPL"][0].date == date(2024, 1, 2)  # YYYYMMDD parsed
    assert by_sym["AAPL"][0].close == 186.0
    assert by_sym["MSFT"][0].close == 372.0


def test_empty_and_junk_bodies_yield_no_bars() -> None:
    assert parse_price_csv("") == []
    assert parse_price_csv("No data") == []
    assert (
        parse_price_csv("Date,Open,High,Low,Close,Volume\n", default_symbol="X") == []
    )  # header only


def test_read_file_and_dir(tmp_path: Path) -> None:
    (tmp_path / "aapl.us.csv").write_text(DL_CSV)
    (tmp_path / "bulk.txt").write_text(BULK_TXT)

    aapl = read_price_csv(tmp_path / "aapl.us.csv")  # symbol inferred from the name
    assert {b.symbol for b in aapl} == {"AAPL"}

    files = iter_price_files(tmp_path)
    assert len(files) == 2  # both CSV/TXT files under the dir
    bars = read_price_files(files)
    assert {b.symbol for b in bars} == {"AAPL", "MSFT"}  # d/l aapl + bulk aapl/msft


def test_single_file_symbol_override(tmp_path: Path) -> None:
    f = tmp_path / "weird-name.csv"
    f.write_text(DL_CSV)
    bars = read_price_files([f], symbol="TSLA")  # override applies to a single per-symbol file
    assert {b.symbol for b in bars} == {"TSLA"}
