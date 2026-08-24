"""API 路由注册"""

from fastapi import FastAPI

from app.api.agent import router as agent_router


def register_routers(app: FastAPI) -> None:
    """注册所有 API 路由"""
    app.include_router(agent_router)
