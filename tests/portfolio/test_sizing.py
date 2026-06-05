"""The unified book sizer: governed equal-weight baseline + breadth-preserving tilt."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET, synthetic_price_observations
from market_trader.core.time import DISTANT_FUTURE
from market_trader.portfolio.risk import RiskLimits
from market_trader.portfolio.sizing import size_book
from market_trader.storage import InMemoryBitemporalStore


def _returns(n_syms: int = 8, n_days: int = 120) -> pd.DataFrame:
    syms = [f"S{i}" for i in range(n_syms)]
    obs = synthetic_price_observations(symbols=syms, start=date(2022, 1, 3), n_days=n_days, seed=3)
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))
    return panel.pct_change().dropna()


def test_size_book_no_scores_is_governed_equal_weight() -> None:
    rets = _returns()
    limits = RiskLimits(max_position_weight=0.5, max_gross_exposure=1.0, max_net_exposure=1.0)

    w = size_book(rets, target_vol=0.10, limits=limits)  # no scores -> the deploy baseline

    assert set(w) == set(rets.columns)  # holds the whole universe (full breadth)
    assert all(v > 0 for v in w.values())  # long-only
    vals = list(w.values())
    assert max(vals) - min(vals) < 1e-9  # equal weight (nothing to tilt on)
    assert sum(abs(v) for v in w.values()) <= 1.0 + 1e-9  # gross cap respected


def test_size_book_tilt_is_long_only_breadth_preserving_and_capped() -> None:
    rets = _returns()
    syms = list(rets.columns)
    scores = pd.Series(np.linspace(-1.0, 1.0, len(syms)), index=syms)  # ascending conviction
    limits = RiskLimits(max_position_weight=0.30, max_gross_exposure=1.0, max_net_exposure=1.0)

    w = size_book(rets, target_vol=0.10, limits=limits, scores=scores, tilt_strength=1.0)

    assert set(w) == set(syms)  # breadth preserved: every name still held
    assert all(v > 0 for v in w.values())  # exp tilt is always positive -> long-only, no clip
    assert max(abs(v) for v in w.values()) <= 0.30 + 1e-9  # per-name cap
    assert w[syms[-1]] > w[syms[0]]  # the top-scored name outweighs the bottom


def test_size_book_regime_derisk_halves_gross() -> None:
    rets = _returns()
    # Caps far from binding, so only the vol target governs gross.
    limits = RiskLimits(
        max_position_weight=100.0, max_gross_exposure=1000.0, max_net_exposure=1000.0
    )

    full = size_book(rets, target_vol=0.10, limits=limits, regime_derisk=1.0)
    derisked = size_book(rets, target_vol=0.10, limits=limits, regime_derisk=0.5)

    full_gross = sum(abs(v) for v in full.values())
    derisked_gross = sum(abs(v) for v in derisked.values())
    assert full_gross > 0
    assert abs(derisked_gross - 0.5 * full_gross) < 1e-9  # halving the vol budget halves gross


def test_size_book_cold_start_falls_back_to_equal_weight() -> None:
    rets = _returns(n_days=10)  # < 20 clean rows -> covariance not estimable
    limits = RiskLimits(max_position_weight=0.5, max_gross_exposure=1.0)

    w = size_book(rets, target_vol=0.10, limits=limits)  # degrades gracefully, still capped

    assert set(w) == set(rets.columns)
    vals = list(w.values())
    assert max(vals) - min(vals) < 1e-9  # ungoverned, but a clean equal book
