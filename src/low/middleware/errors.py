from collections.abc import Awaitable, Callable
import logging
from typing import Self

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.low.exceptions import TimeoutException, resolve_exception
from src.low.logging import logger


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Собирает все ошибки по системе, и конвертирует их в стандартный JSON ответ
    """

    def __init__(self: Self, app: FastAPI) -> None:
        super().__init__(app)
        self.error_logger = logger.register_logger(
            "middleware.errors", level=logging.ERROR
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except TimeoutError:
            exc = TimeoutException()
            error_response = resolve_exception(exc, self.error_logger)
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.model_dump(),
            )
        except Exception as e:
            error_response = resolve_exception(e, self.error_logger)
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.model_dump(),
            )
