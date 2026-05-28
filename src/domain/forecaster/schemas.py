from datetime import date, datetime, timedelta
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, Field, field_validator


def _uci_season(dt: date) -> int:
    """Сезон по разметке UCI Bike Sharing (астрономические границы)."""
    moy = (dt.month, dt.day)
    if (3, 21) <= moy <= (6, 20):
        return 2
    if (6, 21) <= moy <= (9, 22):
        return 3
    if (9, 23) <= moy <= (12, 20):
        return 4
    return 1


class PredictionRequest(BaseModel):
    dteday: date = Field(..., description="Дата наблюдения (YYYY-MM-DD)")
    hr: int = Field(..., ge=0, le=23, description="Час дня (0–23)")
    holiday: Literal[0, 1] = Field(..., description="1 если государственный праздник")
    weathersit: Literal[1, 2, 3, 4] = Field(
        ...,
        description="Погода: 1=ясно, 2=туман, 3=лёгкий дождь, 4=снег",
    )
    temp: float = Field(..., ge=0.0, le=1.0, description="Нормализованная температура")
    hum: float = Field(..., ge=0.0, le=1.0, description="Нормализованная влажность")
    windspeed: float = Field(
        ..., ge=0.0, le=1.0, description="Нормализованная скорость ветра"
    )

    @field_validator("dteday")
    @classmethod
    def validate_date_scope(cls, v: date) -> date:
        dataset_start = date(2011, 1, 1)
        dataset_end = date(2012, 12, 31)
        margin = timedelta(days=365)
        if not (dataset_start - margin <= v <= dataset_end + margin):
            raise ValueError(
                "Дата должна быть в пределах ±1 года от датасета (2011–2012)"
            )
        return v


class PredictionResponse(BaseModel):
    predicted_cnt: float = Field(..., description="Прогноз спроса (оригинальная шкала)")
    cold_start: bool = Field(
        ..., description="True если temporal признаки заполнены из средних по БД"
    )


class TemporalFeatures(BaseModel):
    """Временные признаки в log1p-шкале, возвращаемые репозиторием."""

    cnt_lag_1: float | None = None
    cnt_lag_3: float | None = None
    cnt_lag_6: float | None = None
    cnt_lag_12: float | None = None
    cnt_lag_24: float | None = None
    cnt_rolling_mean_3: float | None = None
    cnt_rolling_mean_6: float | None = None
    cnt_rolling_mean_12: float | None = None
    cnt_rolling_mean_24: float | None = None
    cnt_rolling_std_6: float | None = None
    cnt_rolling_std_12: float | None = None
    cnt_ewm_6h: float | None = None

    @property
    def is_cold(self: Self) -> bool:
        return any(v is None for v in self.model_dump().values())


class FeatureVector(BaseModel):
    """Полный вектор признаков для CatBoost (34 признака, порядок из train.csv)."""

    yr: float
    mnth: float
    hr: float
    holiday: float
    weekday: float
    workingday: float
    temp: float
    hum: float
    windspeed: float
    weather_1: float
    weather_2: float
    weather_3: float
    season_1: float
    season_2: float
    season_3: float
    season_4: float
    hr_sin: float
    hr_cos: float
    mnth_sin: float
    mnth_cos: float
    is_rush_hour: float
    is_night: float
    cnt_lag_1: float
    cnt_lag_3: float
    cnt_lag_6: float
    cnt_lag_12: float
    cnt_lag_24: float
    cnt_rolling_mean_3: float
    cnt_rolling_mean_6: float
    cnt_rolling_mean_12: float
    cnt_rolling_mean_24: float
    cnt_rolling_std_6: float
    cnt_rolling_std_12: float
    cnt_ewm_6h: float

    @classmethod
    def from_request(
        cls, request: PredictionRequest, temporal: TemporalFeatures
    ) -> "FeatureVector":
        dteday = request.dteday
        hr = request.hr
        mnth = dteday.month
        # isoweekday(): Mon=1..Sun=7 → dataset: Sun=0..Sat=6
        weekday = dteday.isoweekday() % 7
        season = _uci_season(dteday)
        workingday = int(weekday in {1, 2, 3, 4, 5} and request.holiday == 0)

        def _nan(v: float | None) -> float:
            return v if v is not None else float("nan")

        return cls(
            yr=float(dteday.year - 2011),
            mnth=float(mnth),
            hr=float(hr),
            holiday=float(request.holiday),
            weekday=float(weekday),
            workingday=float(workingday),
            temp=request.temp,
            hum=request.hum,
            windspeed=request.windspeed,
            weather_1=float(request.weathersit == 1),
            weather_2=float(request.weathersit == 2),
            # weathersit=4 (сильный дождь/снег) объединён с weathersit=3 на этапе FE
            weather_3=float(request.weathersit in (3, 4)),
            season_1=float(season == 1),
            season_2=float(season == 2),
            season_3=float(season == 3),
            season_4=float(season == 4),
            hr_sin=float(np.sin(2 * np.pi * hr / 24)),
            hr_cos=float(np.cos(2 * np.pi * hr / 24)),
            # FE использует (mnth - 1) для янв = 0 градусов
            mnth_sin=float(np.sin(2 * np.pi * (mnth - 1) / 12)),
            mnth_cos=float(np.cos(2 * np.pi * (mnth - 1) / 12)),
            is_rush_hour=float(hr in {7, 8, 9, 17, 18, 19} and workingday == 1),
            is_night=float(0 <= hr <= 5),
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


class FilterBikeReading(BaseModel):
    weekday: int | None = None
    season: int | None = None


class CreateBikeReading(BaseModel):
    reading_dt: datetime
    weekday: int
    season: int
    cnt: float


class UpdateBikeReading(BaseModel):
    pass
