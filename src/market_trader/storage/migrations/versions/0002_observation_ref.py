"""add observation.ref discriminator

Revision ID: 0002_observation_ref
Revises: 0001_initial
Create Date: 2026-06-03

A source-native discriminator so facts sharing bitemporal coordinates (e.g. two
insiders filing on the same dates) do not collapse into one observation.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_observation_ref"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("observations", sa.Column("ref", sa.String(length=256), nullable=True))


def downgrade() -> None:
    op.drop_column("observations", "ref")
