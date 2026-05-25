from pathlib import Path

from catboost import CatBoostRegressor
from dependency_injector import containers, providers
from dependency_injector.providers import Object

from src.domain.common import models
from src.domain.forecaster.repositories import BikeReadingRepository
from src.domain.forecaster.services import ForecastService
from src.low.config.modules import ModulesConfig
from src.low.creds.modules import ModulesCreds
from src.low.database import Database, UnitOfWork


def _load_catboost(path: Path | str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


class Container(containers.DeclarativeContainer):
    # Слой 1: Инфраструктура
    config: Object[ModulesConfig] = providers.Object(object)  # type: ignore
    creds: Object[ModulesCreds] = providers.Object(object)  # type: ignore

    db = providers.Singleton(
        Database, config=config.provided.database, creds=creds.provided.database
    )
    session = providers.ContextLocalSingleton(
        lambda f: f(), db.provided.session_factory
    )

    uow = providers.Factory(UnitOfWork, session=session)

    # Слой 2: Синглтон модели (загружается один раз при старте)
    model = providers.Singleton(_load_catboost, path=config.provided.model.path)

    # Слой 3: Репозитории
    bike_reading_repository = providers.Factory(
        BikeReadingRepository,
        model=models.BikeReading,
        session=session,
    )

    # Слой 4: Сервисы
    forecast_service = providers.Factory(
        ForecastService,
        model_config=config.provided.model,
        model=model,
        bike_reading_repository=bike_reading_repository,
    )


container: Container = Container()


async def setup_container(config: ModulesConfig, creds: ModulesCreds) -> None:
    container.config.override(config)
    container.creds.override(creds)
    container.wire(["src.api.v1.predict"])
