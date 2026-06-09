"""Locked single-look holdout — the anti-snooping guard.

Every other gate here reads the *whole* history every time, so a researcher can re-run it as
data arrives and keep the number that looks good — which is how a small-sample fluke (the
insider ``t=4.4 -> noise`` flip) gets promoted. The fix is a sealed holdout: the most-recent
slice of history is set aside, a signal is confirmed on it **once**, and that first verdict is
written to a ledger and **never overwritten**. Re-running returns the original look (flagged as
sealed) instead of a fresh, shoppable number.

This is the complementary guard to ``multiple_testing`` — that corrects for the *breadth* of
search across signals; this corrects for the *repetition* of search over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import utcnow
from market_trader.storage.bitemporal import BitemporalStore

HOLDOUT_LEDGER_DATASET = "validation.holdout_ledger"


@dataclass(frozen=True)
class HoldoutLook:
    """One sealed confirmation of a signal on the holdout slice."""

    signal: str
    decided_at: datetime
    n_dates: int
    mean_ic: float
    t_stat: float
    passed: bool


def holdout_start(store: BitemporalStore, as_of: datetime, *, frac: float = 0.2) -> datetime | None:
    """The first date of the sealed holdout: the last ``frac`` of the price history. Dates on or
    after this are the holdout (looked at once); everything before is free research data."""
    dates = sorted({o.event_time for o in store.as_of(as_of, dataset=PRICE_DATASET)})
    if len(dates) < 2:
        return None
    cut = int(len(dates) * (1.0 - frac))
    return dates[min(cut, len(dates) - 1)]


def prior_look(
    store: BitemporalStore, signal: str, as_of: datetime | None = None
) -> HoldoutLook | None:
    """The first (sealed) look at ``signal`` on the holdout, if one was already recorded."""
    rows = [
        o
        for o in store.as_of(as_of or utcnow(), dataset=HOLDOUT_LEDGER_DATASET)
        if o.entity_id == signal
    ]
    if not rows:
        return None
    o = min(rows, key=lambda r: r.event_time)  # the FIRST look is the one that counts
    v = o.value
    return HoldoutLook(
        signal=signal,
        decided_at=o.event_time,
        n_dates=int(v.get("n_dates", 0)),
        mean_ic=float(v.get("mean_ic", 0.0)),
        t_stat=float(v.get("t_stat", 0.0)),
        passed=bool(v.get("passed", False)),
    )


def _record(store: BitemporalStore, look: HoldoutLook) -> None:
    store.upsert_many(
        [
            with_deterministic_id(
                Observation(
                    source="validation",
                    dataset=HOLDOUT_LEDGER_DATASET,
                    entity_type="signal",
                    entity_id=look.signal,
                    ref=f"holdout:{look.decided_at.date()}",
                    event_time=look.decided_at,
                    knowledge_time=look.decided_at,
                    value={
                        "n_dates": look.n_dates,
                        "mean_ic": look.mean_ic,
                        "t_stat": look.t_stat,
                        "passed": look.passed,
                    },
                )
            )
        ]
    )


def confirm_on_holdout(store: BitemporalStore, candidate: HoldoutLook) -> tuple[HoldoutLook, bool]:
    """Single-look confirmation. Returns ``(look, is_repeat)``.

    If this signal was already confirmed on the holdout, the **original** look is returned with
    ``is_repeat=True`` and nothing is written — you cannot shop for a better number by re-running.
    Otherwise the candidate is recorded as the sealed first look and returned with
    ``is_repeat=False``.
    """
    existing = prior_look(store, candidate.signal, as_of=candidate.decided_at)
    if existing is not None:
        return existing, True
    _record(store, candidate)
    return candidate, False
