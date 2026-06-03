"""Unit tests for the cost model and turnover."""

from __future__ import annotations

from market_trader.backtest.costs import BasicCostModel, ZeroCostModel, one_way_turnover


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
