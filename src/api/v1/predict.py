from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from src.domain.forecaster.schemas import PredictionRequest, PredictionResponse
from src.domain.forecaster.services import ForecastService
from src.low.container import Container

router = APIRouter()


@router.get(
    path="/health",
    tags=["Service"],
)
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@router.post(
    path="/predict",
    response_model=PredictionResponse,
    tags=["Forecasting"],
)
@inject
async def predict(
    request: PredictionRequest,
    forecast_service: ForecastService = Depends(Provide[Container.forecast_service]),
) -> PredictionResponse:
    """Прогноз почасового спроса на велосипеды."""
    return await forecast_service.predict(request)
