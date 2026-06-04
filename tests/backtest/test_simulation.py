"""Backtest the live composite strategy over history + Monte-Carlo distribution."""

from __future__ import annotations

from datetime import date

import numpy as np

from market_trader.backtest.engine import compare_to_baselines
from market_trader.backtest.pit import StorePriceView
from market_trader.backtest.simulation import block_bootstrap_paths, monte_carlo_report
from market_trader.backtest.strategies import CompositeBacktestStrategy
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.storage import InMemoryBitemporalStore


def _store(symbols: list[str], n_days: int = 200):
    obs = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=3
    )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    return store, sorted({o.event_time for o in obs})


def test_composite_strategy_weights_are_a_valid_book() -> None:
    symbols = [f"S{i}" for i in range(20)]
    store, dates = _store(symbols)
    view = StorePriceView(store, as_of_time=dates[-1], dataset=PRICE_DATASET)
    w = CompositeBacktestStrategy(max_positions=8, top_quantile=0.5).target_weights(view, dates[-1])
    assert 1 <= len(w) <= 8
    assert abs(sum(w.values()) - 1.0) < 1e-9  # fully invested
    assert all(v > 0 for v in w.values())  # long-only


def test_backtest_composite_vs_baselines_runs() -> None:
    symbols = [f"S{i}" for i in range(15)]
    store, dates = _store(symbols)
    summaries = compare_to_baselines(
        store, CompositeBacktestStrategy(max_positions=8), dates[60::5]
    )
    assert {"composite", "equal_weight", "buy_and_hold"} <= set(summaries)
    assert summaries["composite"].n_periods > 0


def test_monte_carlo_report_is_ordered_and_deterministic() -> None:
    rets = np.random.default_rng(0).normal(0.0005, 0.01, size=300)
    rep = monte_carlo_report(rets, n_sims=500, seed=7)
    assert rep == monte_carlo_report(rets, n_sims=500, seed=7)  # deterministic with the seed
    assert rep.total_return_q05 <= rep.total_return_q50 <= rep.total_return_q95
    assert 0.0 <= rep.prob_positive <= 1.0
    assert rep.max_drawdown_q05 <= 0.0  # drawdowns are non-positive


def test_block_bootstrap_shape_and_empty_safe() -> None:
    paths = block_bootstrap_paths(np.arange(50.0), n_sims=10, block=5, seed=1)
    assert paths.shape == (10, 50)
    assert block_bootstrap_paths([], n_sims=10).size == 0
