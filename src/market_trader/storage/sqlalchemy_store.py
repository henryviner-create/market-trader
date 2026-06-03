"""SQLAlchemy-backed bitemporal store.

Runs on **Postgres** (production, with ``pgvector`` for the episodic-memory layer
later) and on **SQLite** (fast local/CI tests, no daemon needed). Timestamps are
persisted as *naive UTC* (see ``DECISIONS.md`` D5) so behaviour is identical
across both engines; the application layer always speaks tz-aware UTC.

The point-in-time filter (``knowledge_time <= k``) runs in SQL; revision
collapsing and ordering reuse the shared helpers in :mod:`bitemporal`, so this
store returns byte-identical results to the in-memory oracle.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from market_trader.core.schema import Observation
from market_trader.core.time import ensure_utc, naive_utc, to_utc_lenient
from market_trader.storage.bitemporal import collapse_latest_revisions, sort_observations


class Base(DeclarativeBase):
    pass


class ObservationRow(Base):
    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    knowledge_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # "metadata" is reserved on Declarative classes; map the attribute to that column.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)

    __table_args__ = (
        Index(
            "ix_obs_knowledge_lookup",
            "knowledge_time",
            "source",
            "dataset",
            "entity_id",
        ),
        Index(
            "ix_obs_logical",
            "source",
            "dataset",
            "entity_id",
            "event_time",
            "revision",
        ),
    )


def _to_row(o: Observation) -> ObservationRow:
    return ObservationRow(
        observation_id=str(o.observation_id),
        source=o.source,
        dataset=o.dataset,
        entity_type=o.entity_type,
        entity_id=o.entity_id,
        event_time=naive_utc(o.event_time),
        knowledge_time=naive_utc(o.knowledge_time),
        revision=o.revision,
        value=dict(o.value),
        metadata_=dict(o.metadata),
        ingested_at=naive_utc(o.ingested_at),
    )


def _to_obs(r: ObservationRow) -> Observation:
    return Observation(
        source=r.source,
        dataset=r.dataset,
        entity_type=r.entity_type,
        entity_id=r.entity_id,
        event_time=to_utc_lenient(r.event_time),
        knowledge_time=to_utc_lenient(r.knowledge_time),
        value=dict(r.value),
        metadata=dict(r.metadata_),
        revision=r.revision,
        observation_id=UUID(r.observation_id),
        ingested_at=to_utc_lenient(r.ingested_at),
    )


class SqlAlchemyBitemporalStore:
    """A :class:`~market_trader.storage.bitemporal.BitemporalStore` over SQLAlchemy."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str, *, echo: bool = False) -> SqlAlchemyBitemporalStore:
        return cls(create_engine(url, echo=echo, future=True))

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_schema(self) -> None:
        """Create tables directly from the ORM metadata (used by tests).

        Production uses Alembic migrations, which additionally enable the
        ``pgvector`` extension on Postgres.
        """
        Base.metadata.create_all(self._engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self._engine)

    def add(self, obs: Observation) -> None:
        self.add_many([obs])

    def add_many(self, observations: Iterable[Observation]) -> None:
        rows = [_to_row(o) for o in observations]
        if not rows:
            return
        with Session(self._engine) as session:
            session.add_all(rows)
            session.commit()

    def upsert_many(self, observations: Iterable[Observation]) -> None:
        """Insert-or-replace by ``observation_id``. Idempotent across backends."""
        rows = [_to_row(o) for o in observations]
        if not rows:
            return
        with Session(self._engine) as session:
            for row in rows:
                session.merge(row)
            session.commit()

    def count(self) -> int:
        with Session(self._engine) as session:
            return int(session.scalar(select(func.count()).select_from(ObservationRow)) or 0)

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
        k = naive_utc(ensure_utc(knowledge_time))
        stmt = select(ObservationRow).where(ObservationRow.knowledge_time <= k)
        if source is not None:
            stmt = stmt.where(ObservationRow.source == source)
        if dataset is not None:
            stmt = stmt.where(ObservationRow.dataset == dataset)
        if entity_type is not None:
            stmt = stmt.where(ObservationRow.entity_type == entity_type)
        if entity_id is not None:
            stmt = stmt.where(ObservationRow.entity_id == entity_id)

        with Session(self._engine) as session:
            observations = [_to_obs(r) for r in session.scalars(stmt)]

        if latest_revision_only:
            observations = collapse_latest_revisions(observations)
        return sort_observations(observations)
