"""Tests for the engine CLI (pure parts; serve/migrate need a live process/DB)."""

from __future__ import annotations

import pytest

from market_trader import __version__
from market_trader.cli import build_parser, check_database, health_payload, main


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
