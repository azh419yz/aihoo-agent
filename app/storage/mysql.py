"""MySQL 数据库操作

异步操作 consultation_sessions 和 consultation_messages 表。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models.domain import ConsultationMessage, ConsultationSession

logger = logging.getLogger(__name__)


class MySQLClient:
    """MySQL 异步操作客户端"""

    def __init__(self):
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker | None = None

    @property
    def engine(self) -> AsyncEngine | None:
        """获取数据库引擎（供外部组件如 TcmMatcher 使用）"""
        return self._engine

    async def connect(self) -> None:
        """建立数据库连接池"""
        self._engine = create_async_engine(
            settings.MYSQL_DSN,
            pool_size=5,
            max_overflow=10,
            echo=settings.DEBUG,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        # 测试连接
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("MySQL 连接成功")

    async def close(self) -> None:
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            logger.info("MySQL 连接已关闭")

    # ==================== 会话操作 ====================

    async def create_session(self, session: ConsultationSession) -> None:
        """创建新会话"""
        async with self._session_factory() as db_session:
            query = text("""
                INSERT INTO consultation_sessions
                    (session_id, patient_id, status, paid,
                     patient_info_collected, patient_info_confirmed,
                     chief_complaint, inquiry_json,
                     diagnosis_json, prescription_json,
                     image_urls, patient_mismatch, mismatch_reason)
                VALUES
                    (:session_id, :patient_id, :status, :paid,
                     :patient_info_collected, :patient_info_confirmed,
                     :chief_complaint, :inquiry_json,
                     :diagnosis_json, :prescription_json,
                     :image_urls, :patient_mismatch, :mismatch_reason)
            """)
            await db_session.execute(query, {
                "session_id": session.session_id,
                "patient_id": session.patient_id,
                "status": session.status,
                "paid": session.paid,
                "patient_info_collected": _to_json(session.patient_info_collected),
                "patient_info_confirmed": _to_json(session.patient_info_confirmed),
                "chief_complaint": session.chief_complaint,
                "inquiry_json": _to_json(session.inquiry_json),
                "diagnosis_json": _to_json(session.diagnosis_json),
                "prescription_json": _to_json(session.prescription_json),
                "image_urls": _to_json(session.image_urls),
                "patient_mismatch": session.patient_mismatch,
                "mismatch_reason": session.mismatch_reason,
            })
            await db_session.commit()

    _JSON_FIELDS = {
        "patient_info_collected", "patient_info_confirmed",
        "inquiry_json", "diagnosis_json", "prescription_json",
        "image_urls",
    }

    async def get_session(self, session_id: str) -> ConsultationSession | None:
        """查询会话（自动反序列化 JSON 字段）"""
        async with self._session_factory() as db_session:
            query = text("""
                SELECT * FROM consultation_sessions WHERE session_id = :session_id
            """)
            result = await db_session.execute(query, {"session_id": session_id})
            row = result.fetchone()
            if row is None:
                return None
            data = dict(row._mapping)
            # 反序列化 JSON 字符串为 Python 对象
            for field in self._JSON_FIELDS:
                if isinstance(data.get(field), str):
                    data[field] = json.loads(data[field])
            return ConsultationSession(**data)

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """更新会话字段"""
        if not updates:
            return
        sets = ", ".join(f"{k} = :{k}" for k in updates)
        query = text(f"""
            UPDATE consultation_sessions
            SET {sets}, updated_at = NOW()
            WHERE session_id = :session_id
        """)
        params = {k: _to_json(v) if isinstance(v, (dict, list)) else v
                  for k, v in updates.items()}
        params["session_id"] = session_id

        async with self._session_factory() as db_session:
            await db_session.execute(query, params)
            await db_session.commit()

    # ==================== 消息操作 ====================

    async def save_message(self, message: ConsultationMessage) -> None:
        """保存对话消息"""
        async with self._session_factory() as db_session:
            query = text("""
                INSERT INTO consultation_messages
                    (session_id, role, content, images)
                VALUES
                    (:session_id, :role, :content, :images)
            """)
            await db_session.execute(query, {
                "session_id": message.session_id,
                "role": message.role,
                "content": message.content,
                "images": _to_json(message.images),
            })
            await db_session.commit()

    async def get_messages(
            self,
            session_id: str,
            limit: int = 50,
            offset: int = 0,
    ) -> list[ConsultationMessage]:
        """获取会话消息列表（自动反序列化 JSON 字段）"""
        async with self._session_factory() as db_session:
            query = text("""
                SELECT * FROM consultation_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
                LIMIT :limit OFFSET :offset
            """)
            result = await db_session.execute(query, {
                "session_id": session_id,
                "limit": limit,
                "offset": offset,
            })
            msgs = []
            for row in result.fetchall():
                data = dict(row._mapping)
                if isinstance(data.get("images"), str):
                    data["images"] = json.loads(data["images"])
                msgs.append(ConsultationMessage(**data))
            return msgs


def _to_json(value: Any) -> str | None:
    """将 Python 对象转换为 JSON 字符串"""
    if value is None:
        return None
    import json
    return json.dumps(value, ensure_ascii=False)


# 全局单例
mysql_client = MySQLClient()
