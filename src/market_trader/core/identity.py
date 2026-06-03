"""Deterministic observation identity, for idempotent ingestion.

Re-running a collector must not create duplicate facts. We derive a stable
``observation_id`` (UUIDv5) from a record's *identity* — the bitemporal
coordinates, not the payload:

    (source, dataset, entity_id, event_time, knowledge_time, revision)

Re-ingesting the same fact yields the same id (an upsert no-ops). A genuine
correction is a new ``revision`` and therefore a new id. The payload is
deliberately excluded so a re-fetch that merely re-parses identical coordinates
self-heals via upsert rather than duplicating.
"""

from __future__ import annotations

from uuid import UUID, uuid5

from market_trader.core.schema import Observation

# Fixed project namespace; do not change (would re-key every existing id).
_NAMESPACE = UUID("d5a7e3c2-1b4f-4e8a-9c6d-2f0b8a1e7c34")


def deterministic_id(obs: Observation) -> UUID:
    key = "|".join(
        (
            obs.source,
            obs.dataset,
            obs.entity_id,
            obs.event_time.isoformat(),
            obs.knowledge_time.isoformat(),
            str(obs.revision),
            obs.ref or "",
        )
    )
    return uuid5(_NAMESPACE, key)


def with_deterministic_id(obs: Observation) -> Observation:
    """Return a copy of ``obs`` whose ``observation_id`` is content-derived."""
    return obs.model_copy(update={"observation_id": deterministic_id(obs)})
