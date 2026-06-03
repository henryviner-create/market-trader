"""Time utilities.

The whole system speaks **one** time language internally: timezone-aware UTC.
Storage persists naive UTC (see ``DECISIONS.md`` D5), and every value crossing the
storage boundary is normalised here so nothing downstream has to think about it.

Two distinct clocks matter (see ``core.schema.Observation``):

* **event time**     — when something was true in the world.
* **knowledge time** — when it first became knowable to *us*.

The point-in-time guarantee the rest of the system relies on is expressed purely
in terms of knowledge time.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

# Re-export so callers use ``from market_trader.core.time import UTC``.
__all__ = [
    "DISTANT_FUTURE",
    "DISTANT_PAST",
    "UTC",
    "day_close",
    "ensure_utc",
    "naive_utc",
    "to_utc_lenient",
    "utcnow",
]

# Sentinels for "as of everything we will ever know" / "before anything".
DISTANT_FUTURE = datetime(9999, 1, 1, tzinfo=UTC)
DISTANT_PAST = datetime(1, 1, 1, tzinfo=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as timezone-aware UTC, rejecting naive datetimes.

    Naive datetimes are rejected at the application boundary on purpose: an
    unlabelled timestamp is the kind of ambiguity that silently becomes
    lookahead bias.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime is not allowed; supply a timezone-aware value")
    return dt.astimezone(UTC)


def to_utc_lenient(dt: datetime) -> datetime:
    """Coerce to UTC, *assuming* UTC for naive inputs.

    Used only when reading back from a store that persists naive UTC; never on
    untrusted external input (use :func:`ensure_utc` there).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def naive_utc(dt: datetime) -> datetime:
    """Convert to UTC and drop the tzinfo, for portable storage on the DB side."""
    return ensure_utc(dt).replace(tzinfo=None)


def utcnow() -> datetime:
    """Current time as timezone-aware UTC."""
    return datetime.now(tz=UTC)


def day_close(d: date, *, hour: int = 21, minute: int = 0) -> datetime:
    """A simple end-of-day knowledge time (default 21:00 UTC ≈ US market close).

    Daily bars become knowable at the close of their own session, so we stamp a
    bar's ``knowledge_time`` here. A decision taken at ``day_close(D)`` may use
    bar ``D`` but nothing later.
    """
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)
