from fastapi import APIRouter

from src.api.v1 import predict

api_router = APIRouter()
api_router.include_router(predict.router)
