"""Deterministic synthetic market data.

Phase 0 ships with no external collectors (those arrive in Phase 1). To exercise
the storage and validation tiers end-to-end we generate a small, *seeded*
universe of daily prices. The generator embeds a faint latent momentum factor so
a momentum strategy has something non-trivial to chew on — but we make **no
claim** that any strategy beats the baselines on this toy data; it exists only to
prove the plumbing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from market_trader.core.schema import Observation
from market_trader.core.time import day_close

PRICE_DATASET = "price.ohlcv"


def business_days(start: date, n: int) -> list[date]:
    """``n`` consecutive Mon-Fri dates starting at or after ``start``."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synthetic_price_observations(
    *,
    symbols: list[str],
    start: date,
    n_days: int,
    seed: int = 7,
    start_price: float = 100.0,
) -> list[Observation]:
    """Generate daily close-price observations for ``symbols`` over ``n_days``.

    Each bar's ``knowledge_time`` is the close of its own session, so a decision
    taken at ``day_close(D)`` can use bar ``D`` and nothing later.
    """
    rng = np.random.default_rng(seed)
    days = business_days(start, n_days)
    n_sym = len(symbols)

    drift = rng.normal(0.0003, 0.0002, n_sym)
    vol = rng.uniform(0.010, 0.020, n_sym)
    factor = rng.normal(0.0, 0.005, len(days))  # latent market/momentum factor
    beta = rng.uniform(-1.0, 1.0, n_sym)

    prices = np.full(n_sym, float(start_price))
    obs: list[Observation] = []
    for i, d in enumerate(days):
        ret = drift + beta * factor[i] + rng.normal(0.0, 1.0, n_sym) * vol
        prices = prices * (1.0 + ret)
        kt = day_close(d)
        for j, sym in enumerate(symbols):
            obs.append(
                Observation(
                    source="synthetic",
                    dataset=PRICE_DATASET,
                    entity_type="equity",
                    entity_id=sym,
                    event_time=kt,
                    knowledge_time=kt,
                    value={"close": float(prices[j]), "ret": float(ret[j])},
                    metadata={"generator": "synthetic_price_observations", "seed": seed},
                )
            )
    return obs
