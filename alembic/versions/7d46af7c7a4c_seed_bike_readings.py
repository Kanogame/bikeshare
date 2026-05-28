"""seed_bike_readings

Revision ID: 7d46af7c7a4c
Revises: a6b284fed409
Create Date: 2026-05-24 20:37:03.269003

Популяет bike_readings из datasets/raw.csv.
cnt хранится в log1p-шкале — temporal признаки в сервисе вычисляются в той же шкале.
Путь к файлу переопределяется через переменную окружения DATASET_PATH.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
import os
from typing import TYPE_CHECKING
import uuid

from alembic import op
import numpy as np
import pandas as pd
import sqlalchemy as sa

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "7d46af7c7a4c"
down_revision: str | Sequence[str] | None = "a6b284fed409"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "bike_readings"

_bike_readings = sa.table(
    _TABLE_NAME,
    sa.column("id", sa.UUID(as_uuid=True)),
    sa.column("reading_dt", sa.DateTime),
    sa.column("weekday", sa.Integer),
    sa.column("season", sa.Integer),
    sa.column("cnt", sa.Float),
    sa.column("created_at", sa.DateTime),
    sa.column("updated_at", sa.DateTime),
)


def upgrade() -> None:
    dataset_path = os.environ.get("DATASET_PATH", "datasets/raw.csv")
    df = pd.read_csv(dataset_path, parse_dates=["dteday"])

    now = datetime.now(UTC).replace(tzinfo=None)

    rows = [
        {
            "id": uuid.uuid4(),
            "reading_dt": datetime.combine(r["dteday"].date(), time(hour=int(r["hr"]))),
            "weekday": int(r["weekday"]),
            "season": int(r["season"]),
            "cnt": float(np.log1p(r["cnt"])),
            "created_at": now,
            "updated_at": now,
        }
        for r in df.to_dict("records")
    ]

    op.bulk_insert(_bike_readings, rows)


def downgrade() -> None:
    op.execute(_bike_readings.delete())
