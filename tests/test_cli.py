"""Tests for the engine CLI (pure parts; serve/migrate need a live process/DB)."""

from __future__ import annotations

import pytest

from market_trader import __version__
from market_trader.cli import (
    _symbols_to_fetch,
    build_parser,
    check_database,
    health_payload,
    main,
)


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_check_database_true_on_sqlite(tmp_path) -> None:
    assert check_database(f"sqlite:///{tmp_path / 'h.db'}") is True


def test_check_database_false_on_unreachable() -> None:
    assert check_database("postgresql+psycopg://x:x@127.0.0.1:1/none", timeout=1) is False


def test_health_payload_reports_db_and_version(tmp_path) -> None:
    payload = health_payload(f"sqlite:///{tmp_path / 'h.db'}")
    assert payload["db"] is True
    assert payload["status"] == "ok"
    assert payload["version"] == __version__


def test_symbols_to_fetch_skips_covered_for_resumable_backfill() -> None:
    universe = ["AAPL", "msft", "NVDA"]
    # case-insensitive skip of names already in the store; survivors keep their casing
    assert _symbols_to_fetch(universe, {"AAPL", "NVDA"}) == ["msft"]
    # nothing covered -> the whole universe is still to do
    assert _symbols_to_fetch(universe, set()) == ["AAPL", "msft", "NVDA"]
    # everything covered -> a re-run is a no-op (coverage complete)
    assert _symbols_to_fetch(universe, {"aapl", "msft", "nvda"}) == []


def test_ingest_filings_parser_accepts_resume_flags() -> None:
    args = build_parser().parse_args(
        ["ingest-filings", "--symbols", "AAPL,MSFT", "--refresh", "--budget", "30"]
    )
    assert args.symbols == "AAPL,MSFT"
    assert args.refresh is True
    assert args.budget == 30.0
    # defaults: full universe, resumable (no refresh), 3y lookback
    d = build_parser().parse_args(["ingest-filings"])
    assert d.symbols == "" and d.refresh is False and d.days == 1095


def test_build_universe_parser_has_screen_knobs() -> None:
    d = build_parser().parse_args(["build-universe"])
    assert d.window == 30 and d.min_price == 5.0 and d.top == 400
    assert d.min_dollar_volume == 2e5 and d.max_dollar_volume == 1e7
