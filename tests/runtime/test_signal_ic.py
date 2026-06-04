"""measure_signal_ic: per-signal out-of-sample IC over history (the signal gate)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET, business_days
from market_trader.core.time import day_close
from market_trader.features.base import Feature
from market_trader.features.technical import Momentum
from market_trader.runtime.signal_ic import measure_signal_ic
from market_trader.storage import InMemoryBitemporalStore
from market_trader.storage.bitemporal import BitemporalStore

_AS_OF = day_close(date(2099, 1, 1))


def _trending_store(n_syms: int = 12, n_days: int = 160, seed: int = 1):
    """Steady distinct per-symbol drifts (+ tiny noise): momentum ranks forward
    returns nearly perfectly, a ground-truth predictive signal (IC -> +1)."""
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.003, 0.003, n_syms)
    prices = np.full(n_syms, 100.0)
    obs: list[Observation] = []
    for d in business_days(date(2022, 1, 3), n_days):
        prices = prices * (1.0 + drifts + rng.normal(0.0, 0.0003, n_syms))
        kt = day_close(d)
        for j in range(n_syms):
            obs.append(
                Observation(
                    source="synthetic",
                    dataset=PRICE_DATASET,
                    entity_type="equity",
                    entity_id=f"S{j}",
                    event_time=kt,
                    knowledge_time=kt,
                    value={"close": float(prices[j])},
                )
            )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    return store, [f"S{j}" for j in range(n_syms)]


class _NoiseFeature(Feature):
    family = "test"
    name = "noise"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        names = list(symbols)
        rng = np.random.default_rng(int(as_of.timestamp()))
        return pd.Series(rng.standard_normal(len(names)), index=names, dtype=float)


def test_measure_signal_ic_detects_a_predictive_signal() -> None:
    store, syms = _trending_store()

    out = measure_signal_ic(store, [Momentum(lookback=20)], syms, _AS_OF, horizon_days=5, every=5)

    assert "mom_20" in out
    r = out["mom_20"]
    assert r.n_dates > 5
    assert r.mean_ic > 0.9  # steady per-symbol drift -> momentum ranks forward returns
    assert r.hit_rate >= 0.9
    assert r.ic_t_stat > 2.0  # significant (inf when perfectly consistent)


def test_measure_signal_ic_noise_signal_is_near_zero() -> None:
    store, syms = _trending_store()

    out = measure_signal_ic(store, [_NoiseFeature()], syms, _AS_OF, horizon_days=5, every=5)

    assert "noise" in out
    assert abs(out["noise"].mean_ic) < 0.3  # uncorrelated with returns -> ~0
    assert abs(out["noise"].ic_t_stat) < 4.0  # not significant
