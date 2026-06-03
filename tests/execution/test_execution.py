"""Execution tier (paper): broker, guardrails, engine loop, reconciliation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Literal

import pytest

from market_trader.config import Settings
from market_trader.core.time import day_close
from market_trader.execution import (
    ExecutionEngine,
    KillSwitchEngaged,
    Order,
    OrderRateLimiter,
    OrderSanityError,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperBroker,
    Position,
    load_execution_audit,
    reconcile,
    sanity_check_order,
)
from market_trader.execution.broker import Account
from market_trader.portfolio import RiskLimits
from market_trader.storage import InMemoryBitemporalStore

T = day_close(date(2023, 1, 3))
T2 = day_close(date(2023, 1, 4))


def _engine(
    broker,
    *,
    mode: Literal["paper", "live"] = "paper",
    live=False,
    ceiling=100_000.0,
    audit=None,
) -> ExecutionEngine:
    settings = Settings(
        execution_mode=mode,
        live_trading_enabled=live,
        capital_ceiling=ceiling,
        max_drawdown_halt=0.2,
        max_orders_per_interval=50,
    )
    return ExecutionEngine(
        broker,
        settings=settings,
        limits=RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0),
        audit_store=audit,
    )


def test_paper_broker_fills_idempotently() -> None:
    broker = PaperBroker({"AAPL": 100.0}, starting_cash=10_000.0)
    filled = broker.submit_order(Order("c1", "AAPL", OrderSide.BUY, 10))
    assert filled.status == OrderStatus.FILLED and filled.filled_qty == 10
    assert broker.get_positions()[0].qty == 10
    assert abs(broker.get_account().cash - 9_000.0) < 1e-9

    broker.submit_order(Order("c1", "AAPL", OrderSide.BUY, 10))  # same id => no double fill
    assert broker.get_positions()[0].qty == 10


def test_sanity_check_rejects_bad_orders() -> None:
    with pytest.raises(OrderSanityError):
        sanity_check_order(Order("a", "X", OrderSide.BUY, 0))
    with pytest.raises(OrderSanityError):
        sanity_check_order(Order("a", "X", OrderSide.BUY, float("nan")))
    with pytest.raises(OrderSanityError):
        sanity_check_order(Order("a", "X", OrderSide.BUY, 1e9))
    with pytest.raises(OrderSanityError):
        far = Order("a", "X", OrderSide.BUY, 10, OrderType.LIMIT, limit_price=200.0)
        sanity_check_order(far, ref_price=100.0)
    sanity_check_order(Order("a", "X", OrderSide.BUY, 10))  # valid: no raise


def test_order_rate_limiter() -> None:
    limiter = OrderRateLimiter(2)
    assert limiter.allow() and limiter.allow()
    assert not limiter.allow()


def test_engine_paper_rebalance_fills_to_target() -> None:
    broker = PaperBroker({"A": 100.0, "B": 50.0}, starting_cash=100_000.0)
    orders = _engine(broker).rebalance({"A": 0.5, "B": 0.5}, {"A": 100.0, "B": 50.0}, as_of=T)
    assert all(o.status == OrderStatus.FILLED for o in orders)
    pos = {p.symbol: p.qty for p in broker.get_positions()}
    assert abs(pos["A"] - 500.0) < 1e-6  # 0.5*100k/100
    assert abs(pos["B"] - 1000.0) < 1e-6  # 0.5*100k/50


def test_engine_refuses_live_when_not_armed() -> None:
    engine = _engine(PaperBroker({"A": 100.0}), mode="live", live=False)
    with pytest.raises(RuntimeError):  # assert_live_allowed fails closed
        engine.rebalance({"A": 0.1}, {"A": 100.0}, as_of=T)


def test_engine_kill_switch_halts() -> None:
    engine = _engine(PaperBroker({"A": 100.0}))
    engine.kill_switch.engage("test")
    with pytest.raises(KillSwitchEngaged):
        engine.rebalance({"A": 0.1}, {"A": 100.0}, as_of=T)


def test_engine_drawdown_breaker_trips_and_halts() -> None:
    broker = PaperBroker({"A": 100.0}, starting_cash=100_000.0)
    engine = _engine(broker)
    engine.rebalance({"A": 1.0}, {"A": 100.0}, as_of=T)  # ~1000 sh; peak equity 100k
    broker.set_price("A", 70.0)  # equity -> ~70k (-30%)
    with pytest.raises(KillSwitchEngaged):
        engine.rebalance({"A": 1.0}, {"A": 70.0}, as_of=T2)
    assert engine.kill_switch.engaged


def test_engine_capital_ceiling_caps_size() -> None:
    broker = PaperBroker({"A": 100.0}, starting_cash=100_000.0)
    _engine(broker, ceiling=1_000.0).rebalance({"A": 1.0}, {"A": 100.0}, as_of=T)
    pos = {p.symbol: p.qty for p in broker.get_positions()}
    assert abs(pos.get("A", 0.0) - 10.0) < 1e-6  # 1.0*1000/100, not 1000 shares


def test_engine_audit_logs_orders() -> None:
    store = InMemoryBitemporalStore()
    broker = PaperBroker({"A": 100.0}, starting_cash=100_000.0)
    _engine(broker, audit=store).rebalance({"A": 0.5}, {"A": 100.0}, as_of=T)
    audit = load_execution_audit(store, T)
    assert len(audit) >= 1
    assert audit[0].value["event"] == "order"


def test_halt_and_flatten_closes_positions() -> None:
    broker = PaperBroker({"A": 100.0}, starting_cash=100_000.0)
    engine = _engine(broker)
    engine.rebalance({"A": 1.0}, {"A": 100.0}, as_of=T)
    assert broker.get_positions()
    engine.halt_and_flatten({"A": 100.0}, as_of=T2)
    assert broker.get_positions() == []
    assert engine.kill_switch.engaged


class _AcctBroker:
    """Minimal broker exposing a fixed Account — for the daily-loss kill tests."""

    def __init__(self, equity: float, last_equity: float) -> None:
        self._acct = Account(equity, equity, equity, last_equity=last_equity)
        self.submitted: list[Order] = []

    def get_account(self) -> Account:
        return self._acct

    def get_positions(self) -> list[Position]:
        return []

    def get_open_orders(self) -> list[Order]:
        return []

    def submit_order(self, order: Order) -> Order:
        self.submitted.append(order)
        return replace(order, status=OrderStatus.FILLED, filled_qty=order.qty)

    def cancel_order(self, client_order_id: str) -> None:
        pass


def test_daily_loss_limit_halts_and_engages_kill_switch() -> None:
    broker = _AcctBroker(equity=93_000.0, last_equity=100_000.0)  # -7% on the day
    settings = Settings(
        execution_mode="paper",
        capital_ceiling=100_000.0,
        max_daily_loss=0.05,
        max_drawdown_halt=0.9,  # high, so the daily-loss rail is what trips
        max_orders_per_interval=50,
    )
    engine = ExecutionEngine(
        broker,
        settings=settings,
        limits=RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0),
    )
    with pytest.raises(KillSwitchEngaged):
        engine.rebalance({"AAPL": 0.5}, {"AAPL": 100.0}, as_of=T)
    assert engine.kill_switch.engaged and not broker.submitted  # halted before any order


def test_daily_loss_limit_off_by_default_lets_a_down_day_trade() -> None:
    broker = _AcctBroker(equity=60_000.0, last_equity=100_000.0)  # -40%, but limit disabled
    orders = _engine(broker, ceiling=100_000.0).rebalance({"AAPL": 0.5}, {"AAPL": 100.0}, as_of=T)
    assert orders and broker.submitted  # default max_daily_loss=0 -> no daily halt


def test_reconcile_detects_divergence() -> None:
    divergences = reconcile(
        {"A": 100.0, "B": 50.0}, [Position("A", 100.0, 10.0), Position("C", 5.0, 1.0)]
    )
    symbols = {d.symbol for d in divergences}
    assert "B" in symbols and "C" in symbols  # missing / unexpected
    assert "A" not in symbols  # matches
