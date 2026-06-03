"""Episodic analog retrieval and the neighbour outcome distribution."""

from __future__ import annotations

import numpy as np

from market_trader.memory import Episode, EpisodicMemory


def _clustered_memory() -> EpisodicMemory:
    mem = EpisodicMemory()
    rng = np.random.default_rng(0)
    for i in range(10):  # cluster near origin, good outcomes, risk_on
        mem.add(
            Episode(f"A{i}", np.array([0.0, 0.0]) + rng.normal(0, 0.1, 2), 0.05, regime="risk_on")
        )
    for i in range(10):  # cluster near (10,10), bad outcomes, risk_off
        mem.add(
            Episode(
                f"B{i}", np.array([10.0, 10.0]) + rng.normal(0, 0.1, 2), -0.05, regime="risk_off"
            )
        )
    return mem


def test_retrieves_nearest_cluster_and_its_outcomes() -> None:
    mem = _clustered_memory()
    neighbours = mem.retrieve(np.array([0.0, 0.0]), k=5)
    assert len(neighbours) == 5
    assert all(e.key.startswith("A") for e, _ in neighbours)

    dist = mem.outcome_distribution(np.array([0.0, 0.0]), k=5)
    assert dist["n"] == 5
    assert dist["mean"] > 0 and dist["share_positive"] == 1.0


def test_regime_filter_restricts_analogs() -> None:
    mem = EpisodicMemory()
    mem.add(Episode("on", np.array([0.0, 0.0]), 0.1, regime="risk_on"))
    mem.add(Episode("off", np.array([0.0, 0.0]), -0.1, regime="risk_off"))
    neighbours = mem.retrieve(np.array([0.0, 0.0]), k=5, regime="risk_off")
    assert [e.key for e, _ in neighbours] == ["off"]
