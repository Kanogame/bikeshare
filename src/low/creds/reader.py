import json
from pathlib import Path

from pydantic import BaseModel

from src.low.creds.modules import ModulesCreds


class CredsReader(BaseModel):
    modules: ModulesCreds

    @classmethod
    def load(cls, creds_path: Path) -> "CredsReader":
        data = json.load(creds_path.open())

        return cls.model_validate(data)
