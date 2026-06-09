"""The event-study runner: detect clusters across history and gate on the permutation null."""

from __future__ import annotations

from datetime import date

import numpy as np

from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.core.schema import Observation
from market_trader.core.synthetic import (
    PRICE_DATASET,
    business_days,
    synthetic_price_observations,
)
from market_trader.core.time import day_close
from market_trader.memory.study_runner import run_event_study, significant_event_types
from market_trader.storage import InMemoryBitemporalStore


def test_runs_on_an_empty_store() -> None:
    assert run_event_study(InMemoryBitemporalStore()) == []  # no price history -> nothing


def test_detects_and_aggregates_an_insider_cluster() -> None:
    syms = [f"S{i}" for i in range(6)]
    obs = synthetic_price_observations(symbols=syms, start=date(2023, 1, 2), n_days=160, seed=5)
    store = InMemoryBitemporalStore()
    store.add_many(obs)

    # 3 insider purchases for S0 on real panel dates in the middle third -> a cluster.
    times = sorted({o.knowledge_time for o in obs if o.entity_id == "S0"})
    for j, kt in enumerate(times[80:95:5]):
        store.add_many(
            [
                Observation(
                    source="edgar",
                    dataset=FORM4_DATASET,
                    entity_type="equity",
                    entity_id="S0",
                    event_time=kt,
                    knowledge_time=kt,
                    value={"is_purchase": True, "insider_name": f"insider{j}"},
                    metadata={},
                )
            ]
        )

    studies = run_event_study(store, step_days=5, post_days=5)
    cluster = next((d for d in studies if d.label == "insider_cluster_buy"), None)
    assert cluster is not None and cluster.n >= 1  # detected and a CAR measured

    # A single cluster with no real post-event drift does NOT clear the permutation gate.
    assert significant_event_types(store, lookback_days=None, n_permutations=100) == {}


def _store_with_clusters(*, drift: float, n_names: int = 8, n_days: int = 200, seed: int = 3):
    """Each name gets one insider cluster, optionally followed by ``drift``/day for 3 days."""
    rng = np.random.default_rng(seed)
    syms = [f"S{i}" for i in range(n_names)]
    days = list(business_days(date(2022, 1, 3), n_days))
    cluster_at = {s: 80 + i * 6 for i, s in enumerate(syms)}  # staggered through the middle
    price = {s: 100.0 for s in syms}
    obs: list[Observation] = []
    for di, d in enumerate(days):
        kt = day_close(d)
        for s in syms:
            r = float(rng.normal(0.0, 0.004))
            if cluster_at[s] < di <= cluster_at[s] + 3:
                r += drift  # the post-event drift the study should (or should not) find
            price[s] *= 1.0 + r
            obs.append(
                Observation(
                    source="synthetic",
                    dataset=PRICE_DATASET,
                    entity_type="equity",
                    entity_id=s,
                    event_time=kt,
                    knowledge_time=kt,
                    value={"close": price[s]},
                )
            )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    for s in syms:
        ci = cluster_at[s]
        for j, d in enumerate(days[ci - 2 : ci + 1]):  # 3 purchases in 3 days -> a cluster
            kt = day_close(d)
            store.add_many(
                [
                    Observation(
                        source="edgar",
                        dataset=FORM4_DATASET,
                        entity_type="equity",
                        entity_id=s,
                        event_time=kt,
                        knowledge_time=kt,
                        value={"is_purchase": True, "insider_name": f"ins{j}"},
                        metadata={},
                    )
                ]
            )
    return store


def test_gate_admits_a_placebo_significant_edge() -> None:
    store = _store_with_clusters(drift=0.02)  # a real +2%/day post-cluster drift
    gate = significant_event_types(
        store, step_days=5, post_days=3, n_permutations=100, lookback_days=None
    )
    assert "insider_cluster_buy" in gate and gate["insider_cluster_buy"].mean_car > 0


def test_gate_rejects_clusters_without_real_drift() -> None:
    store = _store_with_clusters(drift=0.0)  # clusters, but no post-event drift
    gate = significant_event_types(
        store, step_days=5, post_days=3, n_permutations=100, lookback_days=None
    )
    assert gate == {}  # nothing beats the random-anchoring null -> the sleeve stays flat
