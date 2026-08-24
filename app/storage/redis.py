"""Redis 缓存操作

管理会话状态缓存，提供读写接口。
Key 格式: consultation:{session_id}
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 缓存客户端"""

    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """建立 Redis 连接"""
        if self._client is None:
            self._client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
            await self._client.ping()
            logger.info("Redis 连接成功")

    async def close(self) -> None:
        """关闭 Redis 连接"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis 连接已关闭")

    def _session_key(self, session_id: str) -> str:
        return f"consultation:{session_id}"

    async def set_session_field(
            self,
            session_id: str,
            field: str,
            value: Any,
    ) -> None:
        """设置会话的某个字段"""
        if not self._client:
            return
        key = self._session_key(session_id)
        if isinstance(value, (dict, list, bool)):
            value = json.dumps(value, ensure_ascii=False)
        await self._client.hset(key, field, value)
        await self._client.expire(key, settings.REDIS_TTL)

    async def get_session_field(
            self,
            session_id: str,
            field: str,
    ) -> Any | None:
        """获取会话的某个字段"""
        if not self._client:
            return None
        key = self._session_key(session_id)
        value = await self._client.hget(key, field)
        if value is None:
            return None
        # 尝试解析为 JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def get_session_all(self, session_id: str) -> dict[str, Any]:
        """获取会话的全部缓存"""
        if not self._client:
            return {}
        key = self._session_key(session_id)
        data = await self._client.hgetall(key)
        result: dict[str, Any] = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    async def delete_session_field(self, session_id: str, field: str) -> None:
        """删除会话的某个字段"""
        if not self._client:
            return
        key = self._session_key(session_id)
        await self._client.hdel(key, field)

    async def delete_session(self, session_id: str) -> None:
        """删除会话缓存"""
        if not self._client:
            return
        key = self._session_key(session_id)
        await self._client.delete(key)

    async def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        if not self._client:
            return False
        key = self._session_key(session_id)
        return await self._client.exists(key) > 0


# 全局单例
redis_client = RedisClient()
