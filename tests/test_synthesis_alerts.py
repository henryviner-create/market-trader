"""Event alerts and ranked synthesis with the case-against."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from market_trader.alerts import generate_alerts
from market_trader.collectors import CongressTradesCollector, Form4Collector, IngestionGateway
from market_trader.core.time import day_close
from market_trader.memory import Episode, EpisodicMemory
from market_trader.storage import InMemoryBitemporalStore
from market_trader.synthesis import synthesize


def _store_with_events() -> InMemoryBitemporalStore:
    store = InMemoryBitemporalStore()
    gw = IngestionGateway(store)
    gw.ingest(
        Form4Collector().normalize(
            [
                {
                    "issuer_ticker": "AAPL",
                    "insider_name": n,
                    "transaction_code": "P",
                    "transaction_date": "2023-01-10",
                    "filing_date": "2023-01-12",
                }
                for n in ("A", "B", "C")
            ]
        )
    )
    gw.ingest(
        CongressTradesCollector().normalize(
            [
                {
                    "representative": "Leader",
                    "chamber": "house",
                    "ticker": "NVDA",
                    "transaction_type": "buy",
                    "transaction_date": "2023-01-01",
                    "disclosure_date": "2023-02-15",
                    "role": "leadership",
                }
            ]
        )
    )
    return store


def test_generate_alerts_for_cluster_and_leadership() -> None:
    alerts = generate_alerts(
        _store_with_events(),
        day_close(date(2023, 2, 20)),
        cluster_threshold=3,
        cluster_window_days=60,
    )
    kinds = {a.kind for a in alerts}
    assert "insider_cluster_buy" in kinds
    assert "leadership_congress_trade" in kinds


def test_synthesize_ranks_and_includes_case_against() -> None:
    store = _store_with_events()
    scores = pd.Series({"AAPL": 1.5, "NVDA": 0.8, "MSFT": -0.5})

    memory = EpisodicMemory()
    memory.add_many([Episode(f"a{i}", np.array([1.5]), 0.04) for i in range(8)])
    query_vectors = {"AAPL": np.array([1.5])}

    recs = synthesize(
        store,
        day_close(date(2023, 1, 25)),
        scores,
        episodic=memory,
        query_vectors=query_vectors,
        top_n=3,
    )
    assert recs[0].symbol == "AAPL"  # highest score
    top = recs[0]
    assert top.case_against != ""  # disconfirming view always present
    assert 0.0 <= top.confidence <= 1.0
    assert "insider_cluster_buy" in top.events  # corroborating event attached
    assert top.analog.get("n", 0) > 0  # analogs retrieved
