from pathlib import Path

from pydantic import AnyHttpUrl, BaseModel


class GeneralConfig(BaseModel):
    creds: Path
    api_v1_str: str
    project_name: str
    version: str


class FrontendConfig(BaseModel):
    origin: list[AnyHttpUrl]


class NetworkingConfig(BaseModel):
    server_host: str
    server_port: int

    frontend: FrontendConfig


class LoggingConfig(BaseModel):
    console_level: str
    file_level: str
    log_path: Path
    max_log_size_mb: int
