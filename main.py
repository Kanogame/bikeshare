from pathlib import Path
from typing import Self

from src.low.app import App
from src.low.config.reader import ConfigReader
from src.low.creds.reader import CredsReader
from src.low.logging import initialize_logging


class Orag:
    """
    Инициализирует и управляет всеми низкоуровневыми компонентами RAG-сервиса.

    Ответственность:
    - Загрузка конфигурации из config/prod/
    - Инициализация логирования
    - Создание и конфигурирование FastAPI приложения
    - Управление жизненным циклом приложения (lifespan)
    - Запуск uvicorn сервера
    """

    def __init__(self: Self, config_path: Path) -> None:
        self.config_path = config_path
        self.load_configs()
        self.create_logger()
        self.create_app()
        self.app.start_uvicorn()

    def get_config_paths(self: Self) -> tuple[Path, Path]:
        """Получить пути к файлам конфигурации и учетных данных."""
        return (
            self.config_path / Path("config.json"),
            self.config_path / Path("creds.json"),
        )

    def load_configs(self: Self) -> None:
        """Загрузить конфигурацию и учетные данные из JSON файлов."""
        cfg, cr = self.get_config_paths()
        self.config = ConfigReader.load(cfg)
        self.creds = CredsReader.load(cr)

    def create_logger(self: Self) -> None:
        """Инициализировать логгеры для всех компонентов приложения."""
        initialize_logging(self.config.logging)

    def create_app(self) -> None:
        """Создать и сконфигурировать FastAPI приложение"""

        self.app = App(self.config, self.creds)
        self.app.create_app()


if __name__ == "__main__":
    config_path = Path("config/prod")
    orag = Orag(config_path)
