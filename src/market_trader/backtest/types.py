"""Shared types for the harness."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

# A target portfolio: symbol -> weight (fraction of capital; may sum to < 1 = cash).
Weights = dict[str, float]


@runtime_checkable
class PointInTimeView(Protocol):
    """A read-only view of the world as it was knowable at :attr:`as_of`.

    A :class:`Strategy` receives *only* this. By construction it cannot see any
    fact with ``knowledge_time`` after ``as_of``.
    """

    @property
    def as_of(self) -> datetime: ...

    def price_panel(self) -> pd.DataFrame: ...

    def returns_panel(self) -> pd.DataFrame: ...

    def universe(self) -> list[str]: ...


@runtime_checkable
class Strategy(Protocol):
    """Maps a point-in-time view to a target portfolio. Stateless w.r.t. the future."""

    name: str

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights: ...
