"""Cold-start IC bootstrap: weight the composite from offline IC until live grades exist."""

from __future__ import annotations

from datetime import date

import numpy as np

from market_trader.config import Settings
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET, business_days
from market_trader.core.time import DISTANT_FUTURE, day_close
from market_trader.features import FeatureStore
from market_trader.features.technical import Momentum
from market_trader.portfolio import composite_score, equal_weights
from market_trader.runtime.cycle import _measured_signal_ic, _signal_ic_for_weighting
from market_trader.runtime.learning import log_cycle_predictions
from market_trader.storage import InMemoryBitemporalStore

_AS_OF = day_close(date(2099, 1, 1))


def _trending_store(n_syms: int = 12, n_days: int = 160, seed: int = 1) -> InMemoryBitemporalStore:
    """Steady distinct per-symbol drifts: momentum ranks forward returns -> a ground-truth IC."""
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
    return store


def _settings() -> Settings:
    return Settings(scorer="composite", ic_weighting=True, ic_min_abs=0.02)


def test_bootstraps_offline_ic_on_a_cold_start() -> None:
    # No graded predictions yet, but the price history makes momentum predictive. The cold
    # start must surface that *offline* IC instead of returning {} (which is equal-weight).
    store = _trending_store()
    syms = [f"S{j}" for j in range(12)]
    feats = [Momentum(lookback=20)]

    ic = _signal_ic_for_weighting(store, _settings(), feats, syms, _AS_OF, horizon_days=5)

    assert ic.get("mom_20", 0.0) > 0.5  # bootstrapped from history, not an equal-weight cold start


def test_a_noise_only_cold_start_stays_equal_weight() -> None:
    # On a pure random walk no signal clears the gate, so the bootstrap correctly returns {}
    # (equal weight) rather than manufacturing weights from noise.
    rng = np.random.default_rng(0)
    obs: list[Observation] = []
    prices = np.full(8, 100.0)
    for d in business_days(date(2022, 1, 3), 160):
        prices = prices * (1.0 + rng.normal(0.0, 0.01, 8))  # i.i.d. -> momentum has no edge
        kt = day_close(d)
        for j in range(8):
            obs.append(
                Observation(
                    source="synthetic",
                    dataset=PRICE_DATASET,
                    entity_type="equity",
                    entity_id=f"N{j}",
                    event_time=kt,
                    knowledge_time=kt,
                    value={"close": float(prices[j])},
                )
            )
    store = InMemoryBitemporalStore()
    store.add_many(obs)

    ic = _signal_ic_for_weighting(
        store, _settings(), [Momentum(lookback=20)], [f"N{j}" for j in range(8)], _AS_OF
    )

    assert ic == {}  # nothing significant -> equal-weight, no fake alpha from noise


def test_prefers_live_graded_ic_once_it_exists() -> None:
    # Once the scorer has its own matured predictions, use those (the live loop), not the prior.
    store = _trending_store()
    syms = [f"S{j}" for j in range(12)]
    feats = [Momentum(lookback=20)]
    dates = sorted({o.event_time for o in store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET)})
    early, end = dates[-15], dates[-1]  # leave room for the 5-day forward window to mature

    fs = FeatureStore(store, feats)
    matrix = fs.compute_matrix(early, syms)
    scores = composite_score(matrix, equal_weights(matrix.columns))
    log_cycle_predictions(store, scores, matrix, early, model_version="composite")

    live = _measured_signal_ic(store, _settings(), end)
    assert live  # graded predictions produced a live IC
    ic = _signal_ic_for_weighting(store, _settings(), feats, syms, end, horizon_days=5)
    assert ic == live  # used the live graded IC, never reached the offline bootstrap
