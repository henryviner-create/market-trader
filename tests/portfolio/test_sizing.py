"""The unified risk-based sizing pipeline (vol-target + Kelly tilt + hard caps)."""

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


def test_size_book_respects_all_caps() -> None:
    rets = _returns()
    syms = list(rets.columns)
    scores = pd.Series(np.linspace(1.0, -1.0, len(syms)), index=syms)  # mixed long/short
    limits = RiskLimits(max_position_weight=0.25, max_gross_exposure=2.0, max_net_exposure=0.3)

    w = size_book(scores, rets, target_vol=0.10, limits=limits, kelly_fraction=0.25)

    assert w  # non-empty
    assert max(abs(v) for v in w.values()) <= 0.25 + 1e-9  # per-name cap
    assert sum(abs(v) for v in w.values()) <= 2.0 + 1e-9  # gross cap
    assert abs(sum(w.values())) <= 0.3 + 1e-9  # net cap (the previously-unenforced one)


def test_size_book_regime_derisk_shrinks_the_book() -> None:
    rets = _returns()
    syms = list(rets.columns)
    scores = pd.Series(np.linspace(1.0, -1.0, len(syms)), index=syms)
    # Caps far from binding, so only the vol target governs gross.
    limits = RiskLimits(
        max_position_weight=100.0, max_gross_exposure=1000.0, max_net_exposure=1000.0
    )

    full = size_book(scores, rets, target_vol=0.10, limits=limits, regime_derisk=1.0)
    derisked = size_book(scores, rets, target_vol=0.10, limits=limits, regime_derisk=0.5)

    full_gross = sum(abs(v) for v in full.values())
    derisked_gross = sum(abs(v) for v in derisked.values())
    assert full_gross > 0
    assert abs(derisked_gross - 0.5 * full_gross) < 1e-9  # halving the vol budget halves gross
