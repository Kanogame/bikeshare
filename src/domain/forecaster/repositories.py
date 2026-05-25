from datetime import datetime, timedelta
from typing import Self

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.domain.base.base_repository import BaseRepository
from src.domain.common.models import BikeReading
from src.domain.forecaster.schemas import (
    CreateBikeReading,
    FilterBikeReading,
    TemporalFeatures,
    UpdateBikeReading,
)


class BikeReadingRepository(
    BaseRepository[BikeReading, CreateBikeReading, UpdateBikeReading, FilterBikeReading]
):
    """Репозиторий исторических записей аренды велосипедов.

    cnt хранится в log1p-шкале, temporal признаки вычисляются в той же шкале.
    """

    async def get_temporal_features(
        self: Self, reading_dt: datetime
    ) -> TemporalFeatures:
        """Вычислить temporal признаки для момента reading_dt из исторических записей.

        Fetches записи из окна [reading_dt - 25h, reading_dt), вычисляет lag/rolling/EWM.
        """
        since = reading_dt - timedelta(hours=25)
        rows = await self._fetch_window(since, reading_dt)

        if not rows:
            return TemporalFeatures()

        return self._compute_temporal(rows, reading_dt)

    async def _fetch_window(
        self: Self, since: datetime, until: datetime
    ) -> list[BikeReading]:
        result = await self.session.execute(
            sa.select(self.model)
            .where(self.model.reading_dt >= since)
            .where(self.model.reading_dt < until)
            .order_by(self.model.reading_dt.asc())
        )
        return list(result.scalars().all())

    def _compute_temporal(
        self: Self, rows: list[BikeReading], reading_dt: datetime
    ) -> TemporalFeatures:
        """Вычислить lag/rolling/EWM из упорядоченных исторических записей."""
        series = pd.Series(
            {r.reading_dt: r.cnt for r in rows},
            dtype=float,
        ).sort_index()

        def _lag(hours: int) -> float | None:
            target = reading_dt - timedelta(hours=hours)
            if target not in series.index:
                return None
            val = series[target]
            return float(val) if not np.isnan(val) else None

        def _rolling_mean(window: int) -> float | None:
            recent = series[series.index < reading_dt].tail(window)
            if recent.empty:
                return None
            return float(recent.mean())

        def _rolling_std(window: int) -> float | None:
            recent = series[series.index < reading_dt].tail(window)
            if len(recent) < 2:
                return None
            return float(recent.std())

        def _ewm() -> float | None:
            recent = series[series.index < reading_dt]
            if recent.empty:
                return None
            return float(recent.ewm(span=6, adjust=False).mean().iloc[-1])

        return TemporalFeatures(
            cnt_lag_1=_lag(1),
            cnt_lag_3=_lag(3),
            cnt_lag_6=_lag(6),
            cnt_lag_12=_lag(12),
            cnt_lag_24=_lag(24),
            cnt_rolling_mean_3=_rolling_mean(3),
            cnt_rolling_mean_6=_rolling_mean(6),
            cnt_rolling_mean_12=_rolling_mean(12),
            cnt_rolling_mean_24=_rolling_mean(24),
            cnt_rolling_std_6=_rolling_std(6),
            cnt_rolling_std_12=_rolling_std(12),
            cnt_ewm_6h=_ewm(),
        )
