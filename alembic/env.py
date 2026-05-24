import asyncio
import json
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from src.domain.common.models import Base  # noqa: E402

target_metadata = Base.metadata


def _build_url() -> str:
    """Читает URL из файлов конфигурации проекта."""
    cfg_path = Path("config/prod/config.json")
    creds_path = Path("config/prod/creds.json")

    with cfg_path.open() as f:
        cfg = json.load(f)
    with creds_path.open() as f:
        creds = json.load(f)

    db = cfg["modules"]["database"]
    dc = creds["modules"]["database"]
    return (
        f"postgresql+asyncpg://{dc['user']}:{dc['password']}"
        f"@{db['server']}:{db['port']}/{db['db_name']}"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_build_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: object) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)  # type: ignore
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_build_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
