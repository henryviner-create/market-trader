"""Data-freshness monitoring.

"EDGAR quiet 6h" is the kind of alert that catches a broken feed before it
quietly poisons the weighting engine. We track, per source, how long since the
most recent knowable observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from market_trader.core.time import utcnow
from market_trader.storage.bitemporal import BitemporalStore


@dataclass(frozen=True)
class SourceFreshness:
    source: str
    last_knowledge_time: datetime | None
    hours_stale: float | None
    stale: bool


def data_freshness(
    store: BitemporalStore,
    sources: Sequence[str],
    *,
    now: datetime | None = None,
    max_age_hours: float = 24.0,
) -> list[SourceFreshness]:
    now = now or utcnow()
    results: list[SourceFreshness] = []
    for source in sources:
        observations = store.as_of(now, source=source)
        if not observations:
            results.append(SourceFreshness(source, None, None, True))
            continue
        last = max(o.knowledge_time for o in observations)
        hours = (now - last).total_seconds() / 3600.0
        results.append(SourceFreshness(source, last, hours, hours > max_age_hours))
    return results
