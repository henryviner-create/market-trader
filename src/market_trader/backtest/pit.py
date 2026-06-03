"""Point-in-time data views over the bitemporal store.

A :class:`StorePriceView` is the *only* thing a strategy is handed during a
backtest. It queries the store at a fixed ``as_of`` knowledge time, so it
physically cannot surface a price that was not yet knowable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.storage.bitemporal import BitemporalStore


def observations_to_price_frame(
    observations: Iterable[Observation], field_name: str = "close"
) -> pd.DataFrame:
    """Pivot price observations into a ``[event_time x entity_id]`` panel of values."""
    records: list[tuple[datetime, str, float]] = []
    for o in observations:
        v = o.value.get(field_name)
        if v is None:
            continue
        records.append((o.event_time, o.entity_id, float(v)))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records, columns=["event_time", "entity_id", "val"])
    panel = df.pivot_table(index="event_time", columns="entity_id", values="val", aggfunc="last")
    panel = panel.sort_index()
    panel.columns.name = None
    panel.index.name = "event_time"
    return panel


@dataclass
class StorePriceView:
    store: BitemporalStore
    as_of_time: datetime
    dataset: str = PRICE_DATASET
    price_field: str = "close"
    _panel: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @property
    def as_of(self) -> datetime:
        return self.as_of_time

    def price_panel(self) -> pd.DataFrame:
        if self._panel is None:
            obs = self.store.as_of(self.as_of_time, dataset=self.dataset)
            self._panel = observations_to_price_frame(obs, self.price_field)
        return self._panel

    def returns_panel(self) -> pd.DataFrame:
        panel = self.price_panel()
        if panel.empty:
            return panel
        return panel.pct_change().iloc[1:]

    def universe(self) -> list[str]:
        """Symbols with at least one known price at or before ``as_of``."""
        panel = self.price_panel()
        if panel.empty:
            return []
        last = panel.ffill().iloc[-1]
        return [str(c) for c in panel.columns if pd.notna(last[c])]
