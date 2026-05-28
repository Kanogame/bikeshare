"""initial_schema

Revision ID: a6b284fed409
Revises:
Create Date: 2026-05-24 20:37:02.759098

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "a6b284fed409"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bike_readings",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("reading_dt", sa.DateTime, nullable=False),
        sa.Column("weekday", sa.Integer, nullable=False),
        sa.Column("season", sa.Integer, nullable=False),
        sa.Column("cnt", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_bike_readings_reading_dt",
        "bike_readings",
        ["reading_dt"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_bike_readings_reading_dt", table_name="bike_readings")
    op.drop_table("bike_readings")
