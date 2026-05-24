import json
from pathlib import Path

from pydantic import BaseModel

from src.low.config.default_fields import (
    GeneralConfig,
    LoggingConfig,
    NetworkingConfig,
)
from src.low.config.modules import ModulesConfig


class ConfigReader(BaseModel):
    general: GeneralConfig
    networking: NetworkingConfig
    logging: LoggingConfig
    modules: ModulesConfig

    @classmethod
    def load(cls, config_path: Path) -> "ConfigReader":
        data = json.load(config_path.open())

        return cls.model_validate(data)
