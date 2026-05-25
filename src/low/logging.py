import logging
import logging.handlers
from typing import Self

from src.low.config.default_fields import LoggingConfig
from src.low.exceptions import ServerException


class Logger:
    """
    Структурированный логгер для разных компонентов приложения с поддержкой JSON формата
    и ротации файлов.
    """

    def __init__(self: Self) -> None:
        # Корневой логгер, который будет собирать все логи и перенаправлять их через
        # хендлеры
        self.root_logger = logging.getLogger()
        # Все зарегестрированные логгеры приложения
        self.loggers: dict[str, logging.Logger] = {}

    @property
    def default_formatter(self) -> logging.Formatter:
        "Возвращает форматтер по умолчанию для консольного и файлового логирования."

        return logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def configure(
        self,
        config: LoggingConfig,
        formatter: logging.Formatter | None = None,
    ) -> None:
        """
        Конфигурирует систему логирования на основе конфига
        """
        self.config = config

        formatter = formatter or self.default_formatter

        # Создаем директорию для логов, если ее нет
        self.config.log_path.mkdir(parents=True, exist_ok=True)

        self.root_logger.handlers = []

        # Собираем ВСЕ логи, в python logger за это отвечает logging.DEBUG
        self.root_logger.setLevel(logging.DEBUG)

        # Создаем хедлеры (обработчики-перенаправители логов)
        # Таких сейчас 2:
        # Консольный для всех логов с уровнем >= console_level
        # Файловый с ротацией, если размер файла превышает max_file_size_mb
        # (в МБ) для всех логов с уровнем >= file_level
        self.root_logger.addHandler(self.create_console_handler(formatter))
        self.root_logger.addHandler(self.create_file_handler(formatter))

        # Уровни логирования для сторонних библиотек
        # Suppress verbose libraries
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set component levels
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    def create_console_handler(
        self: Self, formatter: logging.Formatter
    ) -> logging.StreamHandler:
        """
        Создает консольный обработчик с заданным уровнем и форматтером

        Грубо говоря, пайпит все логи с уровнем >= level в консоль
        """

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.console_level)

        console_handler.setFormatter(formatter)
        return console_handler

    def create_file_handler(
        self: Self, formatter: logging.Formatter
    ) -> logging.StreamHandler:
        """
        Создает файловый обработчик с ротацией (т.е. автоматическим созданием новых
        файлов при достижении максимального размера) и заданным уровнем и форматтером

        Грубо говоря, пайпит все логи с уровнем >= level в файлы, которых может быть > 1
        """

        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.log_path / "app.log",
            maxBytes=self.config.max_log_size_mb * 1024 * 1024,
            encoding="utf-8",
        )

        file_handler.setLevel(self.config.file_level)
        file_handler.setFormatter(formatter)
        return file_handler

    def register_logger(
        self, component: str, level: int = logging.INFO
    ) -> logging.Logger:
        """
        Зарегистрировать логгер для конкретного компонента приложения с опциональным
        уровнем логирования.

        По умолнчанию, уровень логирования - INFO
        ! Обратить внимание - DEBUG вероятно не будет на консоли
        """

        logger = logging.getLogger(component)
        logger.setLevel(level)

        self.loggers[component] = logger
        return logger

    def get_logger(self, component: str) -> logging.Logger:
        """

        Создать или получить логгер для конкретного компонента приложения.
        Логгеры кэшируются, в состояние объекта класса чтобы обеспечить единообразие и
        избежать дублирования.

        Аргументы:
            component: Навзвание компонента (e.g., "api", "database", "security")

        Возвращает:
            Новый или существующий экземпляр логгера для данного компонента
            !В него можно писать логи, любого уровня, но они могут быть не отображены,
            если уровень слишком мал
        """
        if component not in self.loggers:
            raise ServerException(
                f"Logger for component '{component}' not registered."
                + f" Call register_logger({component}) first."
            )
        return self.loggers[component]


# Глобальный логгер
# Он является глобальным, чтобы его можно было использовать ДО инициализации обвязки
# А также потому, что может использоваться в местах где DI не применим ввиду
# низкого уровня
# Должен быть использован при регистрации кастомных логгеров, или для получения
# уже существующих
logger: Logger = Logger()


def initialize_logging(config: LoggingConfig) -> Logger:
    """
    Инициализация и популяция логера из конфига
    """

    logger.configure(config)
    return logger
