"""News-flow features from the ``news.article`` dataset (GDELT etc.), point-in-time.

``NewsAttention`` is trailing article volume per symbol (news flow / attention) —
robust even when the source carries no per-article tone. ``NewsSentiment`` is the
trailing mean tone where the source provides it. Both read only what was knowable
by ``as_of`` (news is knowable at its seen-date), and absence is a neutral 0 so a
name with no coverage still ranks rather than dropping out.

These are *signals*, not verdicts: they must earn their keep out-of-sample like
any other feature (the learning loop's IC + the forecaster gate decide that).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from math import log1p
from typing import Any

import pandas as pd

from market_trader.collectors.gdelt import NEWS_DATASET
from market_trader.features.base import Feature
from market_trader.storage.bitemporal import BitemporalStore


def _news_by_symbol(
    store: BitemporalStore, as_of: datetime, symbols: Sequence[str], window_days: int
) -> dict[str, list[dict[str, Any]]]:
    cutoff = as_of - timedelta(days=window_days)
    wanted = set(symbols)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in store.as_of(as_of, dataset=NEWS_DATASET):
        if o.entity_id in wanted and o.event_time >= cutoff:
            rows[o.entity_id].append(o.value)
    return rows


class NewsAttention(Feature):
    family = "news"

    def __init__(self, window_days: int = 7) -> None:
        self.window_days = window_days
        self.name = f"news_attention_{window_days}"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        rows = _news_by_symbol(store, as_of, symbols, self.window_days)
        data = {s: log1p(len(rows.get(s, []))) for s in symbols}  # damp heavy news days
        return pd.Series(data).reindex(list(symbols))


class NewsSentiment(Feature):
    family = "news"

    def __init__(self, window_days: int = 7) -> None:
        self.window_days = window_days
        self.name = f"news_tone_{window_days}"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        rows = _news_by_symbol(store, as_of, symbols, self.window_days)

        def mean_tone(vals: list[dict[str, Any]]) -> float:
            tones = [float(v["tone"]) for v in vals if v.get("tone") is not None]
            return sum(tones) / len(tones) if tones else 0.0  # no tone -> neutral

        return pd.Series({s: mean_tone(rows.get(s, [])) for s in symbols}).reindex(list(symbols))


def news_features(window_days: int = 7) -> list[Feature]:
    """The news signal family added to the live feature set when news is enabled."""
    return [NewsAttention(window_days), NewsSentiment(window_days)]
