"""Feature framework.

``Feature.compute(store, as_of, symbols)`` returns a cross-sectional Series
indexed by symbol, read only from what was knowable at ``as_of``. The
``FeatureStore`` stacks features into the matrix the forecasting tier consumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from market_trader.storage.bitemporal import BitemporalStore


class Feature(ABC):
    name: str
    family: str  # "technical" | "fundamental" | "flow" | "news" | "macro" | "alt"

    @abstractmethod
    def compute(
        self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]
    ) -> pd.Series: ...


def cross_sectional_zscore(s: pd.Series) -> pd.Series:
    """Z-score across symbols (skipping NaNs). Flat input maps to zeros."""
    v = s.astype(float)
    sd = v.std(ddof=0, skipna=True)
    if pd.isna(sd) or sd == 0:
        return v * 0.0
    return (v - v.mean(skipna=True)) / sd


class FeatureStore:
    def __init__(self, store: BitemporalStore, features: Sequence[Feature]) -> None:
        self._store = store
        self._features = list(features)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._features]

    def compute_matrix(self, as_of: datetime, symbols: Sequence[str]) -> pd.DataFrame:
        symbols = list(symbols)
        columns = {f.name: f.compute(self._store, as_of, symbols) for f in self._features}
        return pd.DataFrame(columns, index=symbols)


def default_features() -> list[Feature]:
    """The *validated* live set the scorer trades. Each earned its keep out-of-sample.

    A new signal does not go here until it clears the gate (positive, significant OOS IC
    via ``signal-ic`` over ``candidate_features``); that keeps unvalidated signals out of
    live sizing.
    """
    from market_trader.features.flow import CongressLeadershipBuys, InsiderNetBuys
    from market_trader.features.technical import MeanReversion, Momentum, Volatility

    return [
        Momentum(lookback=60),
        MeanReversion(lookback=5),
        Volatility(window=20),
        InsiderNetBuys(window_days=90, opportunistic_only=True),  # opportunistic > raw (CMP 2012)
        CongressLeadershipBuys(window_days=120),
    ]


def candidate_features() -> list[Feature]:
    """``default_features`` plus signals under evaluation, for ``signal-ic`` to measure.

    A candidate must show a positive, significant out-of-sample IC here before it is
    promoted into ``default_features`` (the live scorer) — the "earn its place" gate.
    Most factor candidates (value, PEAD, low-vol, 12-1 momentum) are weak-to-reversed on
    a mega-cap universe (they live in small/mid-caps), so they stay parked here; the raw
    insider signal is kept alongside the promoted opportunistic one to keep tracking the
    comparison.
    """
    from market_trader.features.flow import InsiderNetBuys
    from market_trader.features.fundamental import EarningsSurprise, EarningsYield
    from market_trader.features.llm import LLMNewsSentiment
    from market_trader.features.technical import Momentum, Volatility

    return [
        *default_features(),
        InsiderNetBuys(window_days=90),  # raw, tracked against the promoted opportunistic variant
        Momentum(lookback=252, skip=21),
        Volatility(window=120, low_vol=True),
        EarningsYield(),
        EarningsSurprise(),
        LLMNewsSentiment(),  # Opus-extracted news sentiment — the breadth-factory candidate
    ]


def stack_features() -> list[Feature]:
    """The signals to combine into one multi-signal score (the 'mega-alpha').

    Each earned a positive IC on the small/mid-cap universe (where these premia live);
    they span different families — insider flow, earnings surprise, value, momentum,
    low-vol — so they are roughly uncorrelated, which is what makes stacking them raise
    the information ratio (Fundamental Law: IR = IC x sqrt(breadth)).
    """
    from market_trader.features.flow import InsiderNetBuys
    from market_trader.features.fundamental import EarningsSurprise, EarningsYield
    from market_trader.features.technical import Momentum, Volatility

    return [
        InsiderNetBuys(window_days=90, opportunistic_only=True),
        EarningsSurprise(),
        EarningsYield(),
        Momentum(lookback=252, skip=21),
        Volatility(window=120, low_vol=True),
    ]
