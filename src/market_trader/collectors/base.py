"""Collector base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from market_trader.core.schema import Observation


class Collector(ABC):
    """A source-specific collector.

    Subclasses implement :meth:`normalize` (pure: raw payload -> canonical
    observations with correct ``event_time``/``knowledge_time``) and, where there
    is a live endpoint, their own ``fetch``/``collect``. ``parser_version`` is
    stamped into observation metadata so a future reparse is traceable.
    """

    source: str
    parser_version: int = 1

    @abstractmethod
    def normalize(self, raw: Any) -> list[Observation]:
        """Convert a raw payload into canonical observations."""
