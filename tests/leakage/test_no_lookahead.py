"""Leakage tests at the harness level.

The storage guarantee is necessary but not sufficient: the *engine* must also
never let future data influence a past decision. We prove it by inserting future
data and showing past decisions and overlapping P&L are byte-identical.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_trader.backtest import MomentumStrategy, ZeroCostModel, run_backtest
from market_trader.backtest.pit import StorePriceView
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.storage import InMemoryBitemporalStore


def _split_universe() -> tuple[list, list, list]:
    symbols = [f"S{i}" for i in range(8)]
    full = synthetic_price_observations(
        symbols=symbols, start=date(2022, 1, 3), n_days=120, seed=11
    )
    event_days = sorted({o.event_time for o in full})
    cutoff = event_days[60]
    base = [o for o in full if o.event_time <= cutoff]
    future = [o for o in full if o.event_time > cutoff]
    schedule = event_days[25:56:5]  # all <= cutoff, with forward room
    return base, future, schedule


def test_future_inserts_do_not_change_past_decisions() -> None:
    base, future, schedule = _split_universe()
    strategy = MomentumStrategy(lookback=20)

    without_future = InMemoryBitemporalStore()
    without_future.add_many(base)
    r1 = run_backtest(without_future, strategy, schedule, ZeroCostModel())

    with_future = InMemoryBitemporalStore()
    with_future.add_many(base)
    with_future.add_many(future)
    r2 = run_backtest(with_future, strategy, schedule, ZeroCostModel())

    # Decisions are identical...
    pd.testing.assert_series_equal(r1.turnover, r2.turnover)
    # ...and so is realised P&L on the overlapping dates (r2 simply extends further).
    common = r1.net_returns.index.intersection(r2.net_returns.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(r1.net_returns.loc[common], r2.net_returns.loc[common])


def test_point_in_time_view_excludes_future_prices() -> None:
    base, future, _ = _split_universe()
    cutoff = max(o.event_time for o in base)
    store = InMemoryBitemporalStore()
    store.add_many(base)
    store.add_many(future)

    view = StorePriceView(store, as_of_time=cutoff)
    panel = view.price_panel()
    assert not panel.empty
    assert panel.index.max() <= cutoff
