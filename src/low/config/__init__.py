from pathlib import Path

from src.low.config.reader import ConfigReader


def load_config(path: str) -> ConfigReader:
    return ConfigReader.load(Path(path))
