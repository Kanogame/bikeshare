from collections.abc import Awaitable, Callable
import logging
from typing import Any, Self

from src.low.container import container
from src.low.logging import logger

ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class DatabaseSessionMiddleware:
    def __init__(self: Self, app: ASGIApp) -> None:
        self.app = app
        self.session_logger = logger.register_logger(
            "middleware.database", level=logging.WARNING
        )

    async def __call__(
        self: Self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await self._close_session()

    async def _close_session(self: Self) -> None:
        # AsyncSession ленивая - реального соединения не возникает,
        # пока не выполнен первый запрос. Поэтому close() на
        # неиспользованной session - дешевый no-op
        try:
            session = container.session()
            await session.close()
        except Exception as e:
            self.session_logger.warning(f"Failed to close session: {e}")
        finally:
            # Сбрасываем ContextLocalSingleton, чтобы следующий
            # запрос в этом event loop'е получил свежую session
            container.session.reset()
