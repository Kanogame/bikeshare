from collections.abc import AsyncGenerator
import logging
from typing import Any, Self

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn

from src.api.v1.router import api_router
from src.low.config.reader import ConfigReader
from src.low.container import container, setup_container
from src.low.creds.reader import CredsReader
from src.low.exceptions import UserException
from src.low.logging import logger
from src.low.middleware import (
    DatabaseSessionMiddleware,
    ErrorHandlingMiddleware,
)


class App:
    """Обёртка вокруг создания и настройки FastAPI приложения."""

    def __init__(self: Self, config: ConfigReader, creds: CredsReader) -> None:
        self.config = config
        self.creds = creds

    def create_app(self: Self) -> None:
        self.app = FastAPI(lifespan=self.lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_error_handler(
            request: Any, exc: RequestValidationError
        ) -> JSONResponse:
            message = ""
            try:
                message = exc.errors()[0]["msg"]
            except Exception:
                message = "Parsing error"
            user_error = UserException(message=str(message), status_code=400)
            return JSONResponse(
                status_code=user_error.status_code,
                content=user_error.to_api().model_dump(),
            )

        self._configure_middlewares()
        self._register_routers()
        self._setup_openapi()

    def _setup_openapi(self: Self) -> None:
        openapi_schema: dict[str, Any] = self.app.openapi()
        openapi_schema["tags"] = [
            {
                "name": "Forecasting",
                "description": "Прогнозирование спроса на велосипеды",
            },
            {"name": "Service", "description": "Информация о сервисе"},
        ]
        self.app.openapi_schema = openapi_schema

    @asynccontextmanager
    async def lifespan(self: Self, app: FastAPI) -> AsyncGenerator[None]:
        lifespan_logger = logger.register_logger(
            "bikeshare.lifespan", level=logging.INFO
        )

        await setup_container(self.config.modules, self.creds.modules)
        lifespan_logger.info("DI container initialized")

        app.state.db = container.db()
        lifespan_logger.info("Database engine is ready")

        try:
            container.model()
            lifespan_logger.info("CatBoost model loaded successfully")
        except Exception as exc:
            lifespan_logger.exception("Failed to load CatBoost model: %s", exc)
            raise

        try:
            yield
        finally:
            lifespan_logger.info("Shutting down")
            await app.state.db.close()
            lifespan_logger.info("Application shutdown complete")

    def _configure_middlewares(self: Self) -> None:
        self.app.add_middleware(DatabaseSessionMiddleware)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.networking.frontend.origin,  # type: ignore
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(ErrorHandlingMiddleware)  # type: ignore

    def _register_routers(self: Self) -> None:
        self.app.include_router(api_router, prefix=self.config.general.api_v1_str)

    def start_uvicorn(self: Self) -> None:
        uvicorn_logger = logger.register_logger("uvicorn")
        uvicorn_logger.info(
            f"Starting server on port {self.config.networking.server_port}"
        )
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {
                "uvicorn": {"handlers": [], "level": "INFO", "propagate": True},
                "uvicorn.error": {"handlers": [], "level": "INFO", "propagate": True},
                "uvicorn.access": {"handlers": [], "level": "INFO", "propagate": True},
            },
        }
        uvicorn.run(
            self.app,
            host=self.config.networking.server_host,
            port=self.config.networking.server_port,
            log_level=logging.DEBUG,
            log_config=logging_config,
        )
