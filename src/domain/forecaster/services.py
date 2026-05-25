from datetime import datetime, time
from typing import Self

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from src.domain.base.base_service import BaseService
from src.domain.forecaster.repositories import BikeReadingRepository
from src.domain.forecaster.schemas import (
    FeatureVector,
    PredictionRequest,
    PredictionResponse,
    TemporalFeatures,
)
from src.low.config.modules import ModelConfig
from src.low.logging import logger as app_logger


class ForecastService(BaseService):
    """Сервис прогнозирования спроса на велосипеды."""

    def __init__(
        self: Self,
        model_config: ModelConfig,
        model: CatBoostRegressor,
        bike_reading_repository: BikeReadingRepository,
    ) -> None:
        self.config = model_config
        self.model = model
        self.repo = bike_reading_repository
        self.logger = app_logger.register_logger("forecast_service")

    async def predict(self: Self, request: PredictionRequest) -> PredictionResponse:
        reading_dt = datetime.combine(request.dteday, time(hour=request.hr))

        if self.config.immediate_only:
            temporal = TemporalFeatures()
        else:
            temporal = await self.repo.get_temporal_features(reading_dt)

        features = FeatureVector.from_request(request, temporal)

        df = pd.DataFrame([features.model_dump()])
        raw_pred = float(self.model.predict(df)[0])
        if raw_pred < 0.0:
            self.logger.warning(
                "Model predicted negative log1p value %.4f — clamping to 0", raw_pred
            )
            raw_pred = 0.0
        predicted_cnt = float(np.expm1(raw_pred))

        return PredictionResponse(
            predicted_cnt=predicted_cnt, cold_start=temporal.is_cold
        )
