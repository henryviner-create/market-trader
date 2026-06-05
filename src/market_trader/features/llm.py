"""LLM-derived signal features — read the breadth factory's output point-in-time.

The score lands in the ``llm.signal`` dataset (see ``collectors/llm_features``); this
Feature surfaces the most recent score per name within a trailing window, weighted by the
model's stated confidence so a low-confidence read counts for little. Absence is a neutral
0. Like every signal it is a *candidate* until it earns positive, significant out-of-sample
IC — it is parked in ``candidate_features`` for measurement, not in the live book.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from market_trader.collectors.llm_features import LLM_SIGNAL_DATASET
from market_trader.features.base import Feature
from market_trader.storage.bitemporal import BitemporalStore


class LLMNewsSentiment(Feature):
    family = "news"

    def __init__(self, window_days: int = 10) -> None:
        self.window_days = window_days
        self.name = f"llm_news_sentiment_{window_days}d"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        cutoff = as_of - timedelta(days=self.window_days)
        latest: dict[str, tuple[datetime, float]] = {}
        for o in store.as_of(as_of, dataset=LLM_SIGNAL_DATASET):
            if o.knowledge_time <= cutoff:
                continue
            value = float(o.value.get("score", 0.0)) * float(o.value.get("confidence", 0.0))
            prev = latest.get(o.entity_id)
            if prev is None or o.knowledge_time > prev[0]:
                latest[o.entity_id] = (o.knowledge_time, value)  # most recent read wins
        scores = {sym: sv[1] for sym, sv in latest.items()}
        return pd.Series(scores, dtype=float).reindex(list(symbols)).fillna(0.0)
