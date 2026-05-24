from datetime import date, datetime, timedelta
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    dteday: date = Field(..., description="Дата наблюдения (YYYY-MM-DD)")
    hr: int = Field(..., ge=0, le=23, description="Час дня (0–23)")
    holiday: Literal[0, 1] = Field(..., description="1 если государственный праздник")
    weathersit: Literal[1, 2, 3, 4] = Field(
        ...,
        description="Погода: 1=ясно, 2=туман, 3=лёгкий дождь/снег, 4=сильный дождь/снег",
    )
    temp: float = Field(..., ge=0.0, le=1.0, description="Нормализованная температура")
    hum: float = Field(..., ge=0.0, le=1.0, description="Нормализованная влажность")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Нормализованная скорость ветра")

    @field_validator("dteday")
    @classmethod
    def validate_date_scope(cls, v: date) -> date:
        dataset_start = date(2011, 1, 1)
        dataset_end = date(2012, 12, 31)
        margin = timedelta(days=365)
        if not (dataset_start - margin <= v <= dataset_end + margin):
            raise ValueError("Дата должна быть в пределах ±1 года от датасета (2011–2012)")
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

    def filled_with(self: Self, fallback: float) -> "TemporalFeatures":
        return self.model_copy(
            update={k: fallback for k, v in self.model_dump().items() if v is None}
        )


class FeatureVector(BaseModel):
    """Полный вектор признаков для CatBoost (36 признаков, порядок из train.csv)."""

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
    weather_4: float
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
    discomfort: float
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
