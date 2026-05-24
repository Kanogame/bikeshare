import logging
import math
from datetime import date, datetime, time
from typing import Self

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.domain.base.base_service import BaseService
from src.domain.forecaster.repositories import BikeReadingRepository
from src.domain.forecaster.schemas import (
    FeatureVector,
    PredictionRequest,
    PredictionResponse,
    TemporalFeatures,
)

logger = logging.getLogger(__name__)


def _derive_calendar(dteday: date, hr: int, holiday: int) -> dict:
    """Вывести календарные признаки из даты и часа."""
    yr = dteday.year - 2011
    mnth = dteday.month
    # isoweekday(): Mon=1..Sun=7 → dataset: Sun=0..Sat=6
    weekday = dteday.isoweekday() % 7
    season = (dteday.month - 1) // 3 + 1
    workingday = int(weekday in {1, 2, 3, 4, 5} and holiday == 0)
    return {
        "yr": float(yr),
        "mnth": float(mnth),
        "hr": float(hr),
        "weekday": float(weekday),
        "season": season,
        "workingday": float(workingday),
    }


def _build_feature_vector(
    request: PredictionRequest,
    calendar: dict,
    temporal: TemporalFeatures,
) -> FeatureVector:
    """Собрать полный вектор признаков для CatBoost."""
    hr = request.hr
    mnth = int(calendar["mnth"])
    season = calendar["season"]
    workingday = int(calendar["workingday"])
    weathersit = request.weathersit

    def _nan(v: float | None) -> float:
        return v if v is not None else float("nan")

    return FeatureVector(
        yr=calendar["yr"],
        mnth=calendar["mnth"],
        hr=float(hr),
        holiday=float(request.holiday),
        weekday=calendar["weekday"],
        workingday=calendar["workingday"],
        temp=request.temp,
        hum=request.hum,
        windspeed=request.windspeed,
        # One-hot weather
        weather_1=float(weathersit == 1),
        weather_2=float(weathersit == 2),
        weather_3=float(weathersit == 3),
        weather_4=float(weathersit == 4),
        # One-hot season
        season_1=float(season == 1),
        season_2=float(season == 2),
        season_3=float(season == 3),
        season_4=float(season == 4),
        # Cyclic encoding
        hr_sin=float(np.sin(2 * np.pi * hr / 24)),
        hr_cos=float(np.cos(2 * np.pi * hr / 24)),
        mnth_sin=float(np.sin(2 * np.pi * mnth / 12)),
        mnth_cos=float(np.cos(2 * np.pi * mnth / 12)),
        # Engineered features — из ноутбука
        is_rush_hour=float(hr in {7, 8, 9, 17, 18, 19} and workingday == 1),
        is_night=float(0 <= hr <= 5),
        discomfort=request.temp * request.hum,
        # Temporal features (log1p-шкала, NaN при отсутствии)
        cnt_lag_1=_nan(temporal.cnt_lag_1),
        cnt_lag_3=_nan(temporal.cnt_lag_3),
        cnt_lag_6=_nan(temporal.cnt_lag_6),
        cnt_lag_12=_nan(temporal.cnt_lag_12),
        cnt_lag_24=_nan(temporal.cnt_lag_24),
        cnt_rolling_mean_3=_nan(temporal.cnt_rolling_mean_3),
        cnt_rolling_mean_6=_nan(temporal.cnt_rolling_mean_6),
        cnt_rolling_mean_12=_nan(temporal.cnt_rolling_mean_12),
        cnt_rolling_mean_24=_nan(temporal.cnt_rolling_mean_24),
        cnt_rolling_std_6=_nan(temporal.cnt_rolling_std_6),
        cnt_rolling_std_12=_nan(temporal.cnt_rolling_std_12),
        cnt_ewm_6h=_nan(temporal.cnt_ewm_6h),
    )


class ForecastService(BaseService):
    """Сервис прогнозирования спроса на велосипеды."""

    def __init__(
        self: Self,
        model: CatBoostRegressor,
        bike_reading_repository: BikeReadingRepository,
    ) -> None:
        self.model = model
        self.repo = bike_reading_repository

    async def predict(self: Self, request: PredictionRequest) -> PredictionResponse:
        reading_dt = datetime.combine(request.dteday, time(hour=request.hr))
        temporal = await self.repo.get_temporal_features(reading_dt)

        cold_start = temporal.is_cold
        if cold_start:
            temporal = await self._fill_cold_start(temporal, request, reading_dt)

        calendar = _derive_calendar(request.dteday, request.hr, request.holiday)
        features = _build_feature_vector(request, calendar, temporal)

        df = pd.DataFrame([features.model_dump()])
        raw_pred = float(self.model.predict(df)[0])
        if raw_pred < 0.0:
            logger.warning("Model predicted negative log1p value %.4f — clamping to 0", raw_pred)
            raw_pred = 0.0
        predicted_cnt = float(np.expm1(raw_pred))

        return PredictionResponse(predicted_cnt=predicted_cnt, cold_start=cold_start)

    async def _fill_cold_start(
        self: Self,
        temporal: TemporalFeatures,
        request: PredictionRequest,
        reading_dt: datetime,
    ) -> TemporalFeatures:
        weekday = reading_dt.date().isoweekday() % 7
        season = (reading_dt.month - 1) // 3 + 1
        avg = await self.repo.get_cold_start_avg(request.hr, weekday, season)

        if avg is None:
            logger.warning(
                "Cold start: нет данных для hr=%d weekday=%d season=%d — temporal будут NaN",
                request.hr,
                weekday,
                season,
            )
            fallback = math.nan
        else:
            fallback = float(avg)
            logger.warning(
                "Cold start: заполняем temporal средним %.4f для hr=%d weekday=%d season=%d",
                fallback,
                request.hr,
                weekday,
                season,
            )

        return temporal.filled_with(fallback)
