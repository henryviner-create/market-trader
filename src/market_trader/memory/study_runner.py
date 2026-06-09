"""Run an event study over the store's history — measure post-event drift, don't trade it.

The reactive sleeve must *earn* its place exactly like a signal: an event type is only worth
trading if it shows significant abnormal drift out-of-sample. This walks the price history,
detects events at each step (deduped by entity/type/knowledge-time), and aggregates the
knowledge-time-anchored CAR per event type — the measurement that gates the sleeve, before
a single order is placed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE
from market_trader.memory.event_study import (
    EventOutcomeDistribution,
    PlaceboResult,
    aggregate_event_study,
    placebo_event_study,
)
from market_trader.memory.taxonomy import detect_events
from market_trader.storage.bitemporal import BitemporalStore


def _collect_anchors(
    store: BitemporalStore, *, step_days: int, min_history: int
) -> tuple[dict[str, list[tuple[str, datetime]]], pd.DataFrame]:
    """Detect deduped (entity, knowledge_time) anchors per event type, plus the returns panel."""
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))
    if panel.empty:
        return {}, panel
    returns = panel.pct_change().iloc[1:]
    dates = [d.to_pydatetime() for d in pd.DatetimeIndex(panel.index)]
    schedule = dates[min_history::step_days]

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
) -> list[EventOutcomeDistribution]:
    """Per-event-type post-event CAR over the store's price history, most-events-first.

    Returns one :class:`EventOutcomeDistribution` per detected event type (empty if there is
    no price history or no events). Detection is repeated every ``step_days`` and deduped, so
    a persistent cluster counts once per fresh knowledge-time, not once per scan.
    """
    anchors_by_type, returns = _collect_anchors(store, step_days=step_days, min_history=min_history)
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
) -> list[tuple[EventOutcomeDistribution, PlaceboResult]]:
    """As :func:`run_event_study`, plus a permutation null per type — the sharper gate.

    Pairs each event type's CAR with a placebo p-value (its CAR vs random re-anchoring on the
    same names). A type can be t-significant yet FAIL the placebo, which is the tell that its
    t-stat was inflated by clustering — the guard that catches the insider-mirage class.
    """
    anchors_by_type, returns = _collect_anchors(store, step_days=step_days, min_history=min_history)
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
    threshold: float = 1.96,
    step_days: int = 5,
    post_days: int = 5,
) -> dict[str, EventOutcomeDistribution]:
    """Event types whose post-event drift is significant *and positive* — the tradeable set.

    The gate for the event sleeve: it may open a (long-drift) position only for an event type
    returned here, so the sleeve trades on measured edge rather than on every detected event.
    Empty until there is enough history and enough events to clear the t-stat ``threshold``.
    """
    return {
        d.label: d
        for d in run_event_study(store, step_days=step_days, post_days=post_days)
        if d.significant(threshold) and d.mean_car > 0
    }
