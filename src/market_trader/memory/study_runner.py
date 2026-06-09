"""Run an event study over the store's history — measure post-event drift, don't trade it.

The reactive sleeve must *earn* its place exactly like a signal: an event type is only worth
trading if it shows significant abnormal drift out-of-sample. This walks the price history,
detects events at each step (deduped by entity/type/knowledge-time), and aggregates the
knowledge-time-anchored CAR per event type — the measurement that gates the sleeve, before
a single order is placed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE, utcnow
from market_trader.memory.event_study import (
    EventOutcomeDistribution,
    PlaceboResult,
    aggregate_event_study,
    placebo_event_study,
)
from market_trader.memory.taxonomy import detect_events
from market_trader.storage.bitemporal import BitemporalStore


def _gate_store(store: BitemporalStore, lookback_days: int | None) -> BitemporalStore:
    """A bounded in-memory copy for the (heavy) gate computation.

    The gate runs the placebo over the price history; handed a SQL store with a deep backfill it
    would deserialize the whole price table (300k+ rows) every call — minutes on the live box.
    Load only ``lookback_days`` plus an estimation buffer into memory instead. A no-op for an
    already in-memory store (tests / small data), so behaviour there is unchanged.
    """
    from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore

    if lookback_days is None or not isinstance(store, SqlAlchemyBitemporalStore):
        return store
    from market_trader.storage import InMemoryBitemporalStore

    mem = InMemoryBitemporalStore()
    mem.add_many(store.as_of(DISTANT_FUTURE, since=utcnow() - timedelta(days=lookback_days + 120)))
    return mem


def _collect_anchors(
    store: BitemporalStore, *, step_days: int, min_history: int, lookback_days: int | None = None
) -> tuple[dict[str, list[tuple[str, datetime]]], pd.DataFrame]:
    """Detect deduped (entity, knowledge_time) anchors per event type, plus the returns panel.

    ``lookback_days`` bounds the *detection schedule* to recent history (the panel itself is
    kept whole so the market-model estimation windows are intact) — so the sweep doesn't grind
    over a five-year backfill when only recent events are wanted.
    """
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))
    if panel.empty:
        return {}, panel
    returns = panel.pct_change().iloc[1:]
    dates = [d.to_pydatetime() for d in pd.DatetimeIndex(panel.index)]
    schedule = dates[min_history::step_days]
    if lookback_days is not None and dates:
        cutoff = dates[-1] - timedelta(days=lookback_days)
        schedule = [d for d in schedule if d >= cutoff]

    seen: set[tuple[str, str, datetime]] = set()
    anchors_by_type: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for as_of in schedule:
        for ev in detect_events(store, as_of):
            key = (ev.entity_id, str(ev.event_type), ev.knowledge_time)
            if key in seen:
                continue
            seen.add(key)
            anchors_by_type[str(ev.event_type)].append((ev.entity_id, ev.knowledge_time))
    return anchors_by_type, returns


def run_event_study(
    store: BitemporalStore,
    *,
    step_days: int = 5,
    post_days: int = 5,
    min_history: int = 60,
    lookback_days: int | None = None,
) -> list[EventOutcomeDistribution]:
    """Per-event-type post-event CAR over the store's price history, most-events-first.

    Returns one :class:`EventOutcomeDistribution` per detected event type (empty if there is
    no price history or no events). Detection is repeated every ``step_days`` and deduped, so
    a persistent cluster counts once per fresh knowledge-time, not once per scan.
    """
    anchors_by_type, returns = _collect_anchors(
        store, step_days=step_days, min_history=min_history, lookback_days=lookback_days
    )
    if returns.empty:
        return []
    studies = [
        aggregate_event_study(anchors, returns, label=etype, post=post_days)
        for etype, anchors in anchors_by_type.items()
    ]
    return sorted(studies, key=lambda d: d.n, reverse=True)


def run_event_study_with_placebo(
    store: BitemporalStore,
    *,
    step_days: int = 5,
    post_days: int = 5,
    min_history: int = 60,
    n_permutations: int = 200,
    seed: int = 0,
    lookback_days: int | None = None,
) -> list[tuple[EventOutcomeDistribution, PlaceboResult]]:
    """As :func:`run_event_study`, plus a permutation null per type — the sharper gate.

    Pairs each event type's CAR with a placebo p-value (its CAR vs random re-anchoring on the
    same names). A type can be t-significant yet FAIL the placebo, which is the tell that its
    t-stat was inflated by clustering — the guard that catches the insider-mirage class.
    """
    anchors_by_type, returns = _collect_anchors(
        store, step_days=step_days, min_history=min_history, lookback_days=lookback_days
    )
    if returns.empty:
        return []
    paired = [
        (
            aggregate_event_study(anchors, returns, label=etype, post=post_days),
            placebo_event_study(
                anchors,
                returns,
                label=etype,
                post=post_days,
                n_permutations=n_permutations,
                seed=seed,
            ),
        )
        for etype, anchors in anchors_by_type.items()
    ]
    return sorted(paired, key=lambda dp: dp[0].n, reverse=True)


def significant_event_types(
    store: BitemporalStore,
    *,
    step_days: int = 5,
    post_days: int = 5,
    alpha: float = 0.05,
    n_permutations: int = 200,
    lookback_days: int | None = 1095,
    seed: int = 0,
) -> dict[str, EventOutcomeDistribution]:
    """Event types whose post-event drift beats a permutation null — the tradeable set.

    The gate for the event sleeve. It uses the **placebo** test, not the naive i.i.d. t-stat:
    that t-stat is too *lenient* on a small lucky sample (the inflated insider t=4.4) and too
    *conservative* on noisy clustered events (it rejected the same edge at t=1.92 while the
    permutation null put it at p=0.005). The placebo asks the right question — is the mean CAR
    more extreme than random re-anchoring of the same events — so the sleeve opens a position
    only for an edge that survives it (and is positive-drift). Bounded to the last
    ``lookback_days`` so the gate stays fast on a deep backfill.
    """
    return {
        dist.label: dist
        for dist, plac in run_event_study_with_placebo(
            _gate_store(store, lookback_days),  # window a deep SQL backfill -> seconds, not minutes
            step_days=step_days,
            post_days=post_days,
            n_permutations=n_permutations,
            lookback_days=lookback_days,
            seed=seed,
        )
        if plac.significant(alpha) and dist.mean_car > 0
    }
