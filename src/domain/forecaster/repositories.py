from datetime import datetime
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

# Lag_24/rolling_24 требуют 24 предыдущие позиции; берём запас на случай гэпов.
_HISTORY_FETCH_LIMIT = 48


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

        Берёт последние `_HISTORY_FETCH_LIMIT` строк строго до reading_dt и считает
        lag/rolling/EWM по позициям (как `Series.shift(k)` в FE) — устойчиво к гэпам.
        """
        rows = await self._fetch_recent(reading_dt, _HISTORY_FETCH_LIMIT)

        if not rows:
            return TemporalFeatures()

        return self._compute_temporal(rows, reading_dt)

    async def _fetch_recent(
        self: Self, before: datetime, limit: int
    ) -> list[BikeReading]:
        """Последние `limit` строк строго до `before`,
        возвращаются в хронологическом порядке
        """
        result = await self.session.execute(
            sa.select(self.model)
            .where(self.model.reading_dt < before)
            .order_by(self.model.reading_dt.desc())
            .limit(limit)
        )
        return list(reversed(list(result.scalars().all())))

    def _compute_temporal(
        self: Self, rows: list[BikeReading], reading_dt: datetime
    ) -> TemporalFeatures:
        """Вычислить lag/rolling/EWM из упорядоченных исторических записей.

        Лаг и rolling считаются по позициям строк (как `Series.shift(k)` в FE),
        а не по часам — это даёт паритет с маскированием в notebook 02 на гэпах.
        """
        series = pd.Series(
            {r.reading_dt: r.cnt for r in rows},
            dtype=float,
        ).sort_index()
        recent = series[series.index < reading_dt]

        def _lag(positions: int) -> float | None:
            if len(recent) < positions:
                return None
            val = recent.iloc[-positions]
            return float(val) if not np.isnan(val) else None

        def _rolling_mean(window: int) -> float | None:
            tail = recent.tail(window)
            if tail.empty:
                return None
            return float(tail.mean())

        def _rolling_std(window: int) -> float | None:
            tail = recent.tail(window)
            if len(tail) < 2:
                return None
            return float(tail.std())

        def _ewm() -> float | None:
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
