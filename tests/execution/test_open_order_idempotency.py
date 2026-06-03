"""Open-order awareness makes rebalancing idempotent within a fill window.

The engine must count still-open orders as effective holdings, or a re-run before
the prior order fills sees a zero position and double-submits — the exact bug a
tight intraday loop would hit. Driven by the in-memory paper broker.
"""

from __future__ import annotations

from datetime import UTC, datetime

from market_trader.config import Settings
from market_trader.execution.broker import Order, OrderSide, OrderType
from market_trader.execution.engine import ExecutionEngine
from market_trader.execution.paper_broker import PaperBroker
from market_trader.portfolio.risk import RiskLimits

_AS_OF = datetime(2026, 6, 3, 14, 0, tzinfo=UTC)


def _engine(broker: PaperBroker) -> ExecutionEngine:
    return ExecutionEngine(
        broker,
        settings=Settings(execution_mode="paper", capital_ceiling=20_000.0),
        limits=RiskLimits(max_position_weight=0.10),
    )


def test_paper_broker_reports_resting_limit_as_open() -> None:
    broker = PaperBroker({"AAA": 100.0})
    # A non-marketable buy limit (50 < 100) rests rather than filling.
    broker.submit_order(
        Order("rest", "AAA", OrderSide.BUY, 10.0, order_type=OrderType.LIMIT, limit_price=50.0)
    )
    assert [o.client_order_id for o in broker.get_open_orders()] == ["rest"]
    assert not broker.get_positions()  # nothing filled yet


def test_rebalance_does_not_double_submit_against_open_order() -> None:
    target = {"AAA": 0.05}  # 0.05 * 20_000 / 100 = 10 target shares
    prices = {"AAA": 100.0}

    # Baseline: no open order -> the engine submits the buy.
    fresh = PaperBroker({"AAA": 100.0})
    assert _engine(fresh).rebalance(target, prices, as_of=_AS_OF)  # one order placed

    # With an already-open buy for those same 10 shares, the target is already
    # covered -> the engine must submit nothing.
    pending = PaperBroker({"AAA": 100.0})
    pending.submit_order(
        Order("open", "AAA", OrderSide.BUY, 10.0, order_type=OrderType.LIMIT, limit_price=50.0)
    )
    assert _engine(pending).rebalance(target, prices, as_of=_AS_OF) == []
