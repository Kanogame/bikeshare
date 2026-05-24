from datetime import datetime
import uuid

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.functions import now


class Base(DeclarativeBase):
    """Базовый класс для всех ORM моделей"""

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class AuditMixin:
    """Mixin для трекинга времени создания"""

    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=now()
    )

    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=now(), onupdate=now()
    )


class BikeReading(Base, AuditMixin):
    """Историческая запись аренды велосипедов"""

    __tablename__ = "bike_readings"

    reading_dt: Mapped[datetime] = mapped_column(
        sa.DateTime, nullable=False, index=True, unique=True
    )
    weekday: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    season: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    cnt: Mapped[float] = mapped_column(sa.Float, nullable=False)
