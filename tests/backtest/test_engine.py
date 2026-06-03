"""End-to-end tests for the backtest engine and baseline comparison."""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_trader.backtest import (
    BasicCostModel,
    MomentumStrategy,
    ZeroCostModel,
    run_backtest,
    summaries_to_frame,
)
from market_trader.backtest.engine import compare_to_baselines
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.storage import InMemoryBitemporalStore


def _store_and_schedule() -> tuple[InMemoryBitemporalStore, list]:
    symbols = [f"S{i}" for i in range(6)]
    observations = synthetic_price_observations(
        symbols=symbols, start=date(2022, 1, 3), n_days=80, seed=5
    )
    store = InMemoryBitemporalStore()
    store.add_many(observations)
    event_days = sorted({o.event_time for o in observations})
    schedule = event_days[25:-1:5]
    return store, schedule


def test_backtest_is_deterministic_and_costs_reduce_returns() -> None:
    store, schedule = _store_and_schedule()
    strategy = MomentumStrategy()

    free = run_backtest(store, strategy, schedule, ZeroCostModel())
    costed = run_backtest(store, strategy, schedule, BasicCostModel())
    free_again = run_backtest(store, strategy, schedule, ZeroCostModel())

    assert free.summary.n_periods == costed.summary.n_periods
    assert free.equity_curve.notna().all()
    assert free.net_returns.sum() >= costed.net_returns.sum() - 1e-12  # costs only subtract
    pd.testing.assert_series_equal(free.net_returns, free_again.net_returns)


def test_compare_to_baselines_returns_candidate_and_two_baselines() -> None:
    store, schedule = _store_and_schedule()
    summaries = compare_to_baselines(store, MomentumStrategy(), schedule, BasicCostModel())

    assert set(summaries) == {"momentum", "equal_weight", "buy_and_hold"}
    frame = summaries_to_frame(summaries)
    assert frame.shape[0] == 3
    assert "sharpe" in frame.columns and "max_drawdown" in frame.columns
