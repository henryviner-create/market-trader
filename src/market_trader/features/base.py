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
    """A small, decorrelated starter set. Each must earn its keep out-of-sample."""
    from market_trader.features.flow import CongressLeadershipBuys, InsiderNetBuys
    from market_trader.features.technical import MeanReversion, Momentum, Volatility

    return [
        Momentum(lookback=60),
        MeanReversion(lookback=5),
        Volatility(window=20),
        InsiderNetBuys(window_days=90),
        CongressLeadershipBuys(window_days=120),
    ]
