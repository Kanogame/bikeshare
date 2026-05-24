from collections.abc import AsyncGenerator
from typing import Self

from fastapi.concurrency import asynccontextmanager
from pydantic import PostgresDsn
from sqlalchemy import AsyncAdaptedQueuePool
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, registry

from src.low.config.modules import DatabaseConfig
from src.low.creds.modules import DatabaseCreds


class Database:
    """
    Класс базы данных, основанный на:
    sqlalchemy + asyncio + asyncpg
    """

    cfg: DatabaseConfig
    creds: DatabaseCreds

    _base: registry

    _engine: AsyncEngine | None
    _session_factory: async_sessionmaker[AsyncSession] | None

    def __init__(self: Self, config: DatabaseConfig, creds: DatabaseCreds) -> None:
        self.cfg = config
        self.creds = creds
        self._base = declarative_base()

        self._engine = None
        self._session_factory = None

    @property
    def engine(self) -> AsyncEngine:
        """
        lazy engine
        """

        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """
        lazy session_factory
        """

        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    @property
    def base(self) -> registry:
        return self._base

    def _build_database_url(self: Self) -> str:
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.creds.user,
                password=self.creds.password,
                host=self.cfg.server,
                port=self.cfg.port,
                path=self.cfg.db_name,
            )
        )

    def _create_engine(self: Self) -> AsyncEngine:
        """Создает DB engine по конфигу"""

        engine_kwargs = {
            "echo": False,
            "pool_pre_ping": True,
            "pool_size": self.cfg.pool_size,
            "pool_timeout": 30,
            "poolclass": AsyncAdaptedQueuePool,
        }

        return create_async_engine(self._build_database_url(), **engine_kwargs)

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection]:
        """Создает подключение"""
        async with self.engine.connect() as conn:
            yield conn

    async def close(self) -> None:
        """Закрывает все подключения к бд"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
