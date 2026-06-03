"""The read-only `status` P&L summary: today's change, totals, and win/lose names."""

from __future__ import annotations

from market_trader.cli import _portfolio_summary
from market_trader.execution.broker import Account, Position


def test_portfolio_summary_reports_pnl_and_extremes() -> None:
    account = Account(equity=99_640.0, cash=4_000.0, buying_power=4_000.0, last_equity=100_000.0)
    positions = [
        Position("TSLA", 5.0, 200.0, market_value=975.0, unrealized_pl=-25.0),
        Position("GOOGL", 8.0, 100.0, market_value=840.0, unrealized_pl=40.0),
        Position("SHW", 14.0, 70.0, market_value=966.0, unrealized_pl=-34.0),
    ]
    out = _portfolio_summary(account, positions)

    assert "equity=$99,640.00" in out
    assert "3 position(s)" in out
    assert "-360.00 (-0.36%)" in out  # today's P&L = equity - last_equity
    assert "unrealized P&L: -19.00" in out  # -25 + 40 - 34
    assert "SHW -34" in out.split("worst:")[1].splitlines()[0]  # biggest loser leads "worst"
    assert "GOOGL +40" in out.split("best:")[1].splitlines()[0]  # biggest winner leads "best"


def test_portfolio_summary_handles_empty_book_and_missing_last_equity() -> None:
    out = _portfolio_summary(Account(equity=100_000.0, cash=100_000.0, buying_power=100_000.0), [])
    assert "0 position(s)" in out
    assert "today:" not in out  # no last_equity -> day line is skipped, not faked
    assert "unrealized P&L: +0.00" in out
