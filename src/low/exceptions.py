import logging
from typing import Self

from pydantic import BaseModel, ValidationError


class APIException(BaseModel):
    message: str
    status_code: int


class BaseException(Exception):
    """
    Любая ошибка которая может быть отправлена пользователю

    1. Не факт что это наша ошибка
    2. Не факт что на ней нужно печатать стек и ошибки
    """

    public_message: str
    status_code: int

    def __init__(self: Self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.public_message = message
        self.status_code = status_code

    def to_api(self: Self) -> APIException:
        return APIException(message=self.public_message, status_code=self.status_code)


class PanicException(BaseException):
    """
    Любая ошибка которая должна быть напечатана в логах со стеком.
    Также может быть отправлена пользователю
    """

    message: str

    def __init__(
        self: Self, message: str, public_message: str, status_code: int
    ) -> None:
        super().__init__(public_message, status_code)
        self.message = message
        self.public_message = public_message
        self.status_code = status_code

    def panic(self: Self, logger: logging.Logger) -> None:
        """
        Паникует - т.е. печатает стек и сообщение в логах
        """

        # Печатаем в логах
        logger.error(self.message, exc_info=True)  # noqa: LOG014

    @classmethod
    def from_any(cls: type[Self], any_ex: Exception) -> Self:
        """
        Создает PanicException из любой другой ошибки

        Если это уже PanicException, то просто возвращает ее
        Иначе - создает новую с сообщением str(anyEx)
        """

        if isinstance(any_ex, cls):
            return any_ex
        else:
            return cls(
                message=str(any_ex),
                public_message="INTERNAL_SERVER_ERROR",
                status_code=500,
            )


class UserException(BaseException):
    """
    Базовый класс ошибки пользователя
    Может быть отправлена, не паникует

    Возникает в моменты, когда происходит ошибка, в которой виноват пользователь
    Примеры:
        В запросе присутствует ошибка
        Записи с таким id не существует
        Доступ к записи ограничен
    """

    def __init__(self: Self, message: str, status_code: int = 400) -> None:
        self.public_message = message
        self.status_code = status_code


class ServerException(PanicException):
    """
    Базовый класс ошибки сервера
    Может быть отправлена, паникует

    Возникает в моменты, когда происходит ошибка, в которой виноваты мы
    Примеры:
        База данных не ответила
        Файл не найден

    Отличие данного типа ошибки - пользователь никогда не увидит содержание ошибки

    """

    def __init__(self: Self, message: str, status_code: int = 500) -> None:
        super().__init__(
            message=message,
            public_message="INTERNAL_SERVER_ERROR",
            status_code=status_code,
        )


class TimeoutException(PanicException):
    """
    Ошибка таймаута
    Может быть отправлена, паникует

    Возникает в моменты, когда сервер не успел выполнить запрос за отведенное время
    Примеры:
        База в дедлоке
        Блокирующий внешний апи не отвечает

    Эту ошибку увидят все - и клиент и сервер
    """

    def __init__(self: Self) -> None:
        super().__init__(
            message="Timeout deadline exceeded",
            public_message="Произошла ошибка, попробуйте еще раз",
            status_code=504,
        )


def resolve_exception(any_ex: Exception, ex_logger: logging.Logger) -> APIException:
    """
    Определяет что нужно делать с ошибкой

    Независимо от типа ошибки, вернет APIExceptio
    """

    # Pydantic validation error - wrap as user error
    if isinstance(any_ex, ValidationError):
        ex = UserException(message=str(any_ex), status_code=400)
        return ex.to_api()

    # является ли ошибка системной
    if isinstance(any_ex, BaseException):
        if isinstance(any_ex, PanicException):
            # Правильно паниуем
            any_ex.panic(ex_logger)
            return any_ex.to_api()
        else:
            # Ошибка системная, паниковать не надо
            return any_ex.to_api()
    else:
        # Ошибка не системная
        # Формируем ошибку сервера
        ex = PanicException.from_any(any_ex)
        ex.panic(ex_logger)
        return ex.to_api()


class NotFoundException(UserException):
    """Ресурс не найден"""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message=message, status_code=404)


class DublicateException(UserException):
    """Ошибка повторения ресурса"""

    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(message=message, status_code=409)


class UnauthorizatedException(UserException):
    """Ошибка авторизации, например при плохом токене"""

    def __init__(self) -> None:
        super().__init__(message="Ошибка авторизации", status_code=401)


class ForbiddenException(UserException):
    """Нет доступа к ресурсу"""

    def __init__(self, message: str = "Нет доступа") -> None:
        super().__init__(message=message, status_code=403)


class FileNotFoundException(UserException):
    """Файл не найден"""

    def __init__(self) -> None:
        super().__init__(message="Файл не найден", status_code=404)
