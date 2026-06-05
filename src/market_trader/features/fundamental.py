"""Fundamental features from the SEC XBRL quarterly EPS series.

Two signals with robust out-of-sample evidence and low correlation to insider flow —
exactly the independent breadth the Fundamental Law (IR = IC x sqrt(breadth)) needs:

* ``EarningsYield`` — value: trailing-12-month EPS / price (high = cheap).
* ``EarningsSurprise`` — PEAD: standardized unexpected earnings (SUE) of the latest
  quarter, the drift after an earnings surprise.

Both read only ``knowledge_time <= as_of`` facts, so no figure is used before it filed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from market_trader.collectors.fundamentals import FUNDAMENTAL_DATASET
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.features.base import Feature
from market_trader.storage.bitemporal import BitemporalStore


class EarningsYield(Feature):
    """Value factor: trailing-4-quarter EPS divided by the latest price."""

    family = "fundamental"
    name = "earnings_yield"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        latest_px: dict[str, tuple[datetime, float]] = {}
        for o in store.as_of(as_of, dataset=PRICE_DATASET):
            close = o.value.get("close")
            if close is None:
                continue
            cur = latest_px.get(o.entity_id)
            if cur is None or o.event_time > cur[0]:
                latest_px[o.entity_id] = (o.event_time, float(close))

        eps_hist: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for o in store.as_of(as_of, dataset=FUNDAMENTAL_DATASET):
            eps = o.value.get("eps")
            if eps is not None:
                eps_hist[o.entity_id].append((o.event_time, float(eps)))

        out: dict[str, float] = {}
        for s in symbols:
            quarters = sorted(eps_hist.get(s, []))[-4:]  # trailing twelve months
            px = latest_px.get(s)
            if len(quarters) == 4 and px is not None and px[1] > 0:
                out[s] = sum(e for _, e in quarters) / px[1]
        return pd.Series(out, dtype=float).reindex(list(symbols))


class EarningsSurprise(Feature):
    """PEAD: standardized unexpected earnings (SUE) of the most recent quarter.

    SUE = (EPS_q - EPS_{q-4}) / std of the seasonal differences. Only active while the
    latest announcement is within ``drift_days`` (the post-earnings drift window); names
    whose last report is stale, or with too little history for a stable estimate, get NaN
    (no opinion) rather than a misleading zero.
    """

    family = "fundamental"

    def __init__(self, drift_days: int = 90, min_quarters: int = 8) -> None:
        self.drift_days = drift_days
        self.min_quarters = min_quarters
        self.name = "earnings_surprise"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        cutoff = as_of - timedelta(days=self.drift_days)
        hist: dict[str, list[tuple[datetime, float, datetime]]] = defaultdict(list)
        for o in store.as_of(as_of, dataset=FUNDAMENTAL_DATASET):
            eps = o.value.get("eps")
            if eps is not None:
                hist[o.entity_id].append((o.event_time, float(eps), o.knowledge_time))

        out: dict[str, float] = {}
        for s in symbols:
            rows = sorted(hist.get(s, []))  # by fiscal-quarter end
            if len(rows) < self.min_quarters or rows[-1][2] <= cutoff:
                continue  # too little history, or the latest report is outside the drift window
            eps = [r[1] for r in rows]
            diffs = [eps[i] - eps[i - 4] for i in range(4, len(eps))]  # year-over-year surprises
            if len(diffs) < 3:
                continue
            sd = float(pd.Series(diffs[:-1]).std(ddof=1))  # volatility of *prior* surprises
            if sd > 0:
                out[s] = diffs[-1] / sd  # how surprising the latest quarter is vs history
        return pd.Series(out, dtype=float).reindex(list(symbols))
