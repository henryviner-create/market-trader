"""Hard risk controls: limits, order rejection, circuit-breaker, engine wrap."""

from __future__ import annotations

from datetime import date

import pytest

from market_trader.backtest import EqualWeightStrategy, run_backtest
from market_trader.backtest.pit import StorePriceView
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.core.time import day_close
from market_trader.portfolio import (
    DrawdownCircuitBreaker,
    RiskLimitBreach,
    RiskLimits,
    RiskManagedStrategy,
    apply_risk_limits,
    check_order,
)
from market_trader.storage import InMemoryBitemporalStore


def test_apply_risk_limits_clips_per_name_and_drops_zeros() -> None:
    out = apply_risk_limits({"A": 0.5, "B": 0.5, "C": 0.0}, RiskLimits(max_position_weight=0.1))
    assert max(abs(v) for v in out.values()) <= 0.1 + 1e-9
    assert "C" not in out


def test_apply_risk_limits_scales_to_gross_cap() -> None:
    weights = {s: 0.1 for s in "abcdefgh"}  # gross 0.8
    out = apply_risk_limits(weights, RiskLimits(max_position_weight=0.1, max_gross_exposure=0.5))
    assert abs(sum(abs(v) for v in out.values()) - 0.5) < 1e-9


def test_check_order_rejects_over_limit_positions_and_gross() -> None:
    limits = RiskLimits(max_position_weight=0.1, max_gross_exposure=1.0)
    with pytest.raises(RiskLimitBreach):
        check_order("AAPL", 0.2, {}, limits)  # per-name breach
    with pytest.raises(RiskLimitBreach):
        check_order("AAPL", 0.1, {"MSFT": 0.95}, limits)  # gross breach
    check_order("AAPL", 0.05, {"MSFT": 0.05}, limits)  # within limits: no raise


def test_drawdown_circuit_breaker_trips_and_latches() -> None:
    cb = DrawdownCircuitBreaker(max_drawdown=0.20)
    assert not cb.update(100.0)
    assert not cb.update(90.0)  # -10%
    assert cb.update(75.0)  # -25% -> trip
    assert cb.tripped
    assert cb.update(120.0)  # stays tripped until reset


def test_risk_managed_strategy_clips_in_the_engine() -> None:
    syms = [f"S{i}" for i in range(8)]
    store = InMemoryBitemporalStore()
    store.add_many(
        synthetic_price_observations(symbols=syms, start=date(2022, 1, 3), n_days=60, seed=2)
    )
    days = sorted({o.event_time for o in store.as_of(day_close(date(2099, 1, 1)))})

    managed = RiskManagedStrategy(EqualWeightStrategy(), RiskLimits(max_position_weight=0.05))
    weights = managed.target_weights(StorePriceView(store, days[20]), days[20])
    assert weights  # non-empty
    assert max(abs(v) for v in weights.values()) <= 0.05 + 1e-9  # 1/8 clipped to 0.05

    result = run_backtest(store, managed, days[20:-1:5])
    assert result.summary.n_periods > 0
