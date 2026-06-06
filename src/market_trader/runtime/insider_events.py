"""Insider-cluster event sleeve — the gated decision core.

The event study established that ``insider_cluster_buy`` carries significant, positive
post-event drift (CAR ~+3%, t~4 on our data). This turns that into a *decision*: which fresh
insider clusters the reactive sleeve should open as small, time-boxed positions to ride the
drift — and, critically, **only if the event type has cleared the event-study gate**, so the
sleeve trades measured edge, never every detected cluster.

This module is the pure, testable decision: detect clusters knowable as-of, drop the ones
that are stale (the drift window has already passed), already held, and cap the count; each
entry is time-boxed to an exit. The live execution loop (reserving budget, opening/closing
through the ExecutionEngine, persisting sleeve state — mirroring ``news_sleeve``) is wired on
top of this and is a deliberate, separately-validated follow-up; keeping the *decision* pure
means the gate logic is unit-tested without a broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from market_trader.memory.event_study import EventOutcomeDistribution
from market_trader.memory.taxonomy import EventType, detect_events
from market_trader.storage.bitemporal import BitemporalStore

INSIDER_CLUSTER = str(EventType.INSIDER_CLUSTER_BUY)


@dataclass(frozen=True)
class SleeveEntry:
    """One time-boxed sleeve position to open on a fresh, gated insider cluster."""

    symbol: str
    knowledge_time: datetime  # when the cluster became public (the drift anchor)
    exit_by: datetime  # time-boxed exit (open + hold_days)
    n_buys: int  # cluster size (distinct disclosed purchases)
    expected_car: float  # the gate's measured mean post-event drift, for sizing/info


def insider_cluster_entries(
    store: BitemporalStore,
    as_of: datetime,
    *,
    gate: dict[str, EventOutcomeDistribution],
    hold_days: int = 5,
    max_names: int = 5,
    freshness_days: int = 5,
    cluster_threshold: int = 3,
    cluster_window_days: int = 30,
    held: frozenset[str] = frozenset(),
) -> list[SleeveEntry]:
    """Fresh, gated insider-cluster-buy events to open as time-boxed sleeve positions.

    Returns ``[]`` unless ``insider_cluster_buy`` cleared the gate (present in ``gate`` with a
    significant, positive CAR) — so the sleeve stays flat until the event type has earned its
    place out-of-sample. Among the gated clusters, opens those that became knowable within the
    last ``freshness_days`` (the post-event drift hasn't elapsed yet), are not already
    ``held``, newest first, capped at ``max_names``. Each entry exits after ``hold_days``.
    """
    dist = gate.get(INSIDER_CLUSTER)
    if dist is None or not dist.significant() or dist.mean_car <= 0:
        return []  # the event type has not earned its place -> the sleeve does not trade it

    events = detect_events(
        store,
        as_of,
        cluster_threshold=cluster_threshold,
        cluster_window_days=cluster_window_days,
    )
    clusters = [e for e in events if str(e.event_type) == INSIDER_CLUSTER]
    out: list[SleeveEntry] = []
    for e in sorted(clusters, key=lambda ev: ev.knowledge_time, reverse=True):
        if e.entity_id in held:
            continue  # already in the sleeve -> don't double-open / churn
        if (as_of - e.knowledge_time).days > freshness_days:
            continue  # stale: the drift the gate measured has already happened
        out.append(
            SleeveEntry(
                symbol=e.entity_id,
                knowledge_time=e.knowledge_time,
                exit_by=as_of + timedelta(days=hold_days),
                n_buys=int(e.payload.get("n_buys", 0)),
                expected_car=dist.mean_car,
            )
        )
        if len(out) >= max_names:
            break
    return out
