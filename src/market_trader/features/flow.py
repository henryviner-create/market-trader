"""Flow/positioning features from insider (Form 4) and congressional disclosures.

These count *disclosed* activity within a trailing window (by knowledge time), so
a trade contributes only once it was actually knowable. Absence is a neutral 0,
not missing. Congressional signal is restricted to high-signal roles (leadership /
relevant committee) — backbencher trades are noise.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from market_trader.collectors.congress import CONGRESS_DATASET
from market_trader.collectors.edgar import FORM4_DATASET
from market_trader.core.schema import Observation
from market_trader.features.base import Feature
from market_trader.storage.bitemporal import BitemporalStore


def _routine_insiders(observations: list[Observation]) -> frozenset[str]:
    """Insiders who file in the same calendar month in >= 3 distinct years are 'routine'
    (scheduled, uninformative); everyone else is 'opportunistic' and carries essentially
    all of insider trading's predictive power (Cohen-Malloy-Pomorski 2012, "Decoding Inside
    Information"). Filing month (knowledge_time) is the reliable timing proxy. An insider
    with < 3 years of history can't be routine, so it counts as opportunistic.
    """
    seen: dict[str, dict[int, set[int]]] = {}
    for o in observations:
        name = o.value.get("insider_name")
        if not name:
            continue
        kt = o.knowledge_time
        by_month = seen.setdefault(str(name), {})
        by_month.setdefault(kt.month, set()).add(kt.year)
    return frozenset(
        name for name, by_month in seen.items() if any(len(yrs) >= 3 for yrs in by_month.values())
    )


class InsiderNetBuys(Feature):
    family = "flow"

    def __init__(self, window_days: int = 90, *, opportunistic_only: bool = False) -> None:
        self.window_days = window_days
        self.opportunistic_only = opportunistic_only
        self.name = f"insider_net_buys_{window_days}d" + ("_opp" if opportunistic_only else "")

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        cutoff = as_of - timedelta(days=self.window_days)
        obs = store.as_of(as_of, dataset=FORM4_DATASET)
        routine = _routine_insiders(obs) if self.opportunistic_only else frozenset()
        net: dict[str, float] = defaultdict(float)
        for o in obs:
            if o.knowledge_time <= cutoff:
                continue
            if self.opportunistic_only and str(o.value.get("insider_name")) in routine:
                continue  # scheduled, uninformative trader -> drop
            net[o.entity_id] += 1.0 if o.value.get("is_purchase") else -1.0
        return pd.Series(net, dtype=float).reindex(list(symbols)).fillna(0.0)


class CongressLeadershipBuys(Feature):
    family = "flow"

    def __init__(self, window_days: int = 120) -> None:
        self.window_days = window_days
        self.name = f"congress_lead_buys_{window_days}d"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        cutoff = as_of - timedelta(days=self.window_days)
        buys: dict[str, float] = defaultdict(float)
        for o in store.as_of(as_of, dataset=CONGRESS_DATASET):
            if o.knowledge_time <= cutoff:
                continue
            if o.metadata.get("high_signal_role") and o.value.get("transaction_type") == "buy":
                buys[o.entity_id] += 1.0
        return pd.Series(buys, dtype=float).reindex(list(symbols)).fillna(0.0)
