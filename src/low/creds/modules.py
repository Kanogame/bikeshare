from pydantic import BaseModel


class DatabaseCreds(BaseModel):
    user: str
    password: str


class ModulesCreds(BaseModel):
    database: DatabaseCreds
