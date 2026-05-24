from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession


class UnitOfWork:
    """
    Паттерн Unit Of Work
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session: AsyncSession = session

    async def __aenter__(self: Self) -> Self:
        if not self.session.in_transaction():
            await self.begin()
        return self

    async def __aexit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
        await self.close()

    async def begin(self: Self) -> None:
        await self.session.begin()

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def flush(self) -> None:
        await self.session.flush()

    async def close(self) -> None:
        await self.session.close()
