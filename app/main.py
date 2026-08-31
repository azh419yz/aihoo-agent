"""FastAPI 应用入口

AI 中医助手 Agent 服务 - FastAPI 应用启动和配置。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import register_routers
from app.common.handlers import register_exception_handlers
from app.core.config import settings
from app.knowledge.tcm_matcher import tcm_matcher
from app.storage.mysql import mysql_client
from app.storage.redis import redis_client

# 日志配置
_LOG_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# 控制台日志
logging.basicConfig(level=_LOG_LEVEL, format=_LOG_FORMAT)

# 文件日志（自动创建 logs 目录）
_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)

_file_handler = RotatingFileHandler(
    filename=os.path.join(_log_dir, "agent.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setLevel(_LOG_LEVEL)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

# 添加到根日志器
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 %s v%s 启动中...", settings.APP_NAME, settings.APP_VERSION)

    # 启动时连接数据库和缓存
    try:
        await redis_client.connect()
        logger.info("✅ Redis 连接成功")
    except Exception as e:
        logger.warning("⚠️ Redis 连接失败（服务将继续运行）: %s", e)

    try:
        await mysql_client.connect()
        logger.info("✅ MySQL 连接成功")
    except Exception as e:
        logger.warning("⚠️ MySQL 连接失败（服务将继续运行）: %s", e)

    # 启动时加载中医辨病辨证数据
    if mysql_client.engine:
        try:
            await tcm_matcher.load(mysql_client.engine)
            logger.info(
                "✅ TCM 数据加载成功（疾病 %d 条，证型 %d 条）",
                tcm_matcher.disease_count,
                tcm_matcher.syndrome_count,
            )
        except Exception as e:
            logger.warning("⚠️ TCM 数据加载失败（服务将继续运行）: %s", e)

    yield

    # 关闭时清理资源
    await redis_client.close()
    await mysql_client.close()
    logger.info("👋 服务已关闭")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI 中医助手 Agent 服务 - 智能问诊、辨病辨证、处方开具",
    lifespan=lifespan,
)

# CORS 配置（允许 Java 后端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册全局异常处理器
register_exception_handlers(app)

# 注册路由
register_routers(app)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}
