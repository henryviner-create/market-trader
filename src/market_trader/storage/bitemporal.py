"""The bitemporal store interface, its in-memory reference implementation, and the
revision/ordering semantics that every backend must share.

``collapse_latest_revisions`` and ``sort_observations`` are deliberately factored
out so the in-memory and SQL stores produce *byte-identical* results — the
in-memory store can then serve as the oracle the SQL store is tested against.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from market_trader.core.schema import LogicalKey, Observation
from market_trader.core.time import ensure_utc


def collapse_latest_revisions(rows: Iterable[Observation]) -> list[Observation]:
    """Keep, per :attr:`Observation.logical_key`, the most recent revision.

    "Most recent" is ordered by ``(knowledge_time, revision)`` so that, as of a
    given knowledge time, a reader sees the latest correction that was knowable
    by then and never one published later.
    """
    best: dict[LogicalKey, Observation] = {}
    for o in rows:
        cur = best.get(o.logical_key)
        if cur is None or (o.knowledge_time, o.revision) > (cur.knowledge_time, cur.revision):
            best[o.logical_key] = o
    return list(best.values())


def sort_observations(rows: list[Observation]) -> list[Observation]:
    """Stable, deterministic ordering for reproducible reads."""
    return sorted(
        rows,
        key=lambda o: (
            o.knowledge_time,
            o.event_time,
            o.source,
            o.dataset,
            o.entity_id,
            o.revision,
        ),
    )


@runtime_checkable
class BitemporalStore(Protocol):
    """A point-in-time fact store.

    The contract that matters: :meth:`as_of` never returns an observation whose
    ``knowledge_time`` is after the requested ``knowledge_time``.
    """

    def add(self, obs: Observation) -> None: ...

    def add_many(self, observations: Iterable[Observation]) -> None: ...

    def as_of(
        self,
        knowledge_time: datetime,
        *,
        source: str | None = None,
        dataset: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        latest_revision_only: bool = True,
    ) -> list[Observation]: ...

    def count(self) -> int: ...


class InMemoryBitemporalStore:
    """Reference implementation. Simple, obviously-correct, and the test oracle."""

    def __init__(self) -> None:
        self._obs: list[Observation] = []

    def add(self, obs: Observation) -> None:
        self._obs.append(obs)

    def add_many(self, observations: Iterable[Observation]) -> None:
        self._obs.extend(observations)

    def count(self) -> int:
        return len(self._obs)

    def as_of(
        self,
        knowledge_time: datetime,
        *,
        source: str | None = None,
        dataset: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        latest_revision_only: bool = True,
    ) -> list[Observation]:
        k = ensure_utc(knowledge_time)
        rows = [o for o in self._obs if o.knowledge_time <= k]
        if source is not None:
            rows = [o for o in rows if o.source == source]
        if dataset is not None:
            rows = [o for o in rows if o.dataset == dataset]
        if entity_type is not None:
            rows = [o for o in rows if o.entity_type == entity_type]
        if entity_id is not None:
            rows = [o for o in rows if o.entity_id == entity_id]
        if latest_revision_only:
            rows = collapse_latest_revisions(rows)
        return sort_observations(rows)
