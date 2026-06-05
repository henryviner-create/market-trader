"""Price/technical features. All read prices via the point-in-time view."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from market_trader.backtest.pit import StorePriceView
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.features.base import Feature
from market_trader.storage.bitemporal import BitemporalStore

# Features read whichever price dataset they are pointed at, so the same maths run
# on daily bars (default) or on the minute dataset for intraday signals — there a
# ``lookback`` counts minutes, not days.


class Momentum(Feature):
    family = "technical"

    def __init__(self, lookback: int = 60, *, skip: int = 0, dataset: str = PRICE_DATASET) -> None:
        self.lookback = lookback
        self.skip = skip  # exclude the most recent `skip` periods (12-1 momentum: skip=21)
        self.dataset = dataset
        self.name = f"mom_{lookback}" + (f"_skip{skip}" if skip else "")

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        panel = StorePriceView(store, as_of, dataset=self.dataset).price_panel()
        return self._from_panel(panel, symbols)

    def _from_panel(self, panel: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
        """Compute from a precomputed price panel — so a caller iterating many dates
        can slice one panel instead of re-querying/re-pivoting the store per date."""
        if panel.empty or panel.shape[0] < self.lookback + 1:
            return pd.Series(index=list(symbols), dtype=float)
        p = panel.ffill()
        # Return from `lookback` ago up to `skip` ago. skip=0 is plain momentum; the
        # academically robust form skips the most recent month (252-day, skip=21) to
        # avoid the short-term reversal that contaminates raw 12-month momentum.
        mom = p.iloc[-1 - self.skip] / p.iloc[-1 - self.lookback] - 1.0
        return mom.reindex(list(symbols))


class MeanReversion(Feature):
    family = "technical"

    def __init__(self, lookback: int = 5, *, dataset: str = PRICE_DATASET) -> None:
        self.lookback = lookback
        self.dataset = dataset
        self.name = f"meanrev_{lookback}"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        panel = StorePriceView(store, as_of, dataset=self.dataset).price_panel()
        return self._from_panel(panel, symbols)

    def _from_panel(self, panel: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
        if panel.empty or panel.shape[0] < self.lookback + 1:
            return pd.Series(index=list(symbols), dtype=float)
        p = panel.ffill()
        short_ret = p.iloc[-1] / p.iloc[-1 - self.lookback] - 1.0
        return (-short_ret).reindex(list(symbols))  # recent losers favoured


class Volatility(Feature):
    family = "technical"

    def __init__(
        self, window: int = 20, *, low_vol: bool = False, dataset: str = PRICE_DATASET
    ) -> None:
        self.window = window
        self.low_vol = low_vol  # True -> low-volatility factor (rank calm names high)
        self.dataset = dataset
        self.name = f"{'lowvol' if low_vol else 'vol'}_{window}"

    def compute(self, store: BitemporalStore, as_of: datetime, symbols: Sequence[str]) -> pd.Series:
        panel = StorePriceView(store, as_of, dataset=self.dataset).price_panel()
        return self._from_panel(panel, symbols)

    def _from_panel(self, panel: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
        if panel.empty:
            return pd.Series(index=list(symbols), dtype=float)
        returns = panel.pct_change().iloc[1:]
        if returns.empty:
            return pd.Series(index=list(symbols), dtype=float)
        vol = returns.tail(self.window).std(ddof=0)
        # The low-volatility anomaly: calm stocks outperform risk-adjusted, so the
        # *factor* is -vol (low vol -> high score). Plain vol keeps the raw measure.
        return (-vol if self.low_vol else vol).reindex(list(symbols))
