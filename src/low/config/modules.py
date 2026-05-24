from pathlib import Path

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    server: str
    port: int
    db_name: str
    pool_size: int


class ModelConfig(BaseModel):
    path: Path


class ModulesConfig(BaseModel):
    database: DatabaseConfig
    model: ModelConfig
