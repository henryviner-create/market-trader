"""Intraday (minute) price bars -> point-in-time observations.

The daily :class:`~market_trader.collectors.prices.PriceCollector` stamps every
bar at its session close, which is correct for daily data but would collapse all
of a day's minute bars onto one timestamp. The live loop instead lands minute
bars in :data:`PRICE_INTRADAY_DATASET` with their exact minute as ``event_time``
(= ``knowledge_time``: a bar is knowable once its minute has closed), so the same
point-in-time machinery and features work unchanged at minute resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_INTRADAY_DATASET
from market_trader.core.time import ensure_utc

_OPTIONAL_FIELDS = ("open", "high", "low", "volume")


def _parse_ts(raw: Any) -> datetime:
    # Alpaca returns RFC3339 with a trailing "Z"; normalise to UTC-aware.
    return ensure_utc(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))


def intraday_bars_to_observations(
    records: Iterable[dict[str, Any]], *, source: str = "price"
) -> list[Observation]:
    """Map minute-bar records (with ``timestamp``) to intraday-dataset observations."""
    out: list[Observation] = []
    for r in records:
        close = r.get("close")
        if close is None:
            continue
        ts = _parse_ts(r["timestamp"])
        value: dict[str, Any] = {"close": float(close)}
        for field_name in _OPTIONAL_FIELDS:
            v = r.get(field_name)
            if v is not None:
                value[field_name] = float(v)
        out.append(
            Observation(
                source=source,
                dataset=PRICE_INTRADAY_DATASET,
                entity_type="equity",
                entity_id=str(r["symbol"]).upper(),
                event_time=ts,
                knowledge_time=ts,
                value=value,
                metadata={"timeframe": "intraday"},
            )
        )
    return out
