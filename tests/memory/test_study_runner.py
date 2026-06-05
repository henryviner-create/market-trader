"""The event-study runner: detect clusters across history and aggregate their drift."""

from __future__ import annotations

from datetime import date

from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.core.schema import Observation
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.memory.study_runner import run_event_study
from market_trader.storage import InMemoryBitemporalStore


def test_runs_on_an_empty_store() -> None:
    assert run_event_study(InMemoryBitemporalStore()) == []  # no price history -> nothing


def test_detects_and_aggregates_an_insider_cluster() -> None:
    syms = [f"S{i}" for i in range(6)]
    obs = synthetic_price_observations(symbols=syms, start=date(2023, 1, 2), n_days=160, seed=5)
    store = InMemoryBitemporalStore()
    store.add_many(obs)

    # 3 insider purchases for S0 on real panel dates in the middle third -> a cluster whose
    # anchors line up exactly with the returns index.
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
    assert cluster is not None and cluster.n >= 1  # the cluster was detected and a CAR measured
