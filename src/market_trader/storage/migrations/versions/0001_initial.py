"""initial bitemporal schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-03

Creates the canonical ``observations`` table and enables the ``pgvector``
extension (Postgres only) that the episodic-memory layer will use in Phase 3.
Timestamps are stored as naive UTC by design (see DECISIONS.md D5).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "observations",
        sa.Column("observation_id", sa.String(length=36), primary_key=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("dataset", sa.String(length=128), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=False), nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=False), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=False), nullable=False),
    )
    op.create_index(
        "ix_obs_knowledge_lookup",
        "observations",
        ["knowledge_time", "source", "dataset", "entity_id"],
    )
    op.create_index(
        "ix_obs_logical",
        "observations",
        ["source", "dataset", "entity_id", "event_time", "revision"],
    )


def downgrade() -> None:
    op.drop_index("ix_obs_logical", table_name="observations")
    op.drop_index("ix_obs_knowledge_lookup", table_name="observations")
    op.drop_table("observations")
