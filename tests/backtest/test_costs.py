"""Unit tests for the cost model and turnover."""

from __future__ import annotations

from market_trader.backtest.costs import (
    BasicCostModel,
    BorrowCostModel,
    ZeroCostModel,
    one_way_turnover,
)


def test_one_way_turnover() -> None:
    assert one_way_turnover({}, {"A": 1.0}) == 1.0
    assert one_way_turnover({"A": 1.0}, {"B": 1.0}) == 2.0  # both legs traded
    assert one_way_turnover({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}) == 0.0
    assert abs(one_way_turnover({"A": 0.5}, {"A": 0.3}) - 0.2) < 1e-12


def test_basic_cost_model_charges_bps_on_turnover() -> None:
    model = BasicCostModel(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=1.0)  # 4 bps
    # full switch => turnover 2.0 => cost = 2.0 * 4e-4
    assert abs(model.turnover_cost({"A": 1.0}, {"B": 1.0}) - 2.0 * 4e-4) < 1e-15
    assert model.turnover_cost({"A": 1.0}, {"A": 1.0}) == 0.0


def test_zero_cost_model() -> None:
    assert ZeroCostModel().turnover_cost({"A": 1.0}, {"B": 1.0}) == 0.0


def test_basic_and_zero_models_carry_no_holding_cost() -> None:
    assert BasicCostModel().holding_cost({"A": -1.0}, days=365) == 0.0
    assert ZeroCostModel().holding_cost({"A": -1.0}, days=365) == 0.0


def test_borrow_cost_model_charges_on_short_notional() -> None:
    model = BorrowCostModel(annual_borrow_bps=50.0)
    assert model.holding_cost({"A": 0.5, "B": 0.5}, days=365) == 0.0  # long-only: no borrow
    # 0.5 short notional, full year at 50 bps -> 0.5 * 50e-4
    assert abs(model.holding_cost({"A": 0.5, "B": -0.5}, days=365) - 0.5 * 50e-4) < 1e-12
    # half a year -> half the fee
    assert abs(model.holding_cost({"A": -0.4}, days=182) - 0.4 * 50e-4 * (182 / 365)) < 1e-12
    # turnover still behaves like the basic model (inherited 4 bps default)
    assert abs(model.turnover_cost({"A": 1.0}, {"B": 1.0}) - 2.0 * 4e-4) < 1e-15
