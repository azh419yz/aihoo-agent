"""会话管理服务"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.agent.state_machine import SessionState
from app.models.domain import ConsultationSession
from app.storage.mysql import mysql_client
from app.storage.redis import redis_client

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """当前 UTC 时间的 ISO 格式字符串（会话元数据用）"""
    return datetime.now(timezone.utc).isoformat()


# Redis session key prefix
SESSION_KEY = "session"
STATE_KEY = "state"
MEMORY_KEY = "memory"
PATIENT_INFO_KEY = "patient_info"
CHIEF_COMPLAINT_KEY = "chief_complaint"
DIAGNOSIS_KEY = "diagnosis"
PRESCRIPTION_KEY = "prescription"
INQUIRY_KEY = "inquiry"

# 图 final state 中可写回 Redis/MySQL 的白名单（LangGraph 重构后使用）。
# request_* 为请求输入，messages 由 save_message 追加，均不在此列。
SESSION_FIELD_NAMES: list[str] = [
    "state",
    "paid",
    "patient_info",
    "patient_info_confirmed",
    "chief_complaint",
    "inquiry",
    "diagnosis",
    "prescription",
    "image_urls",
    "tongue_analysis",
    "face_analysis",
    "patient_mismatch",
    "mismatch_reason",
    "med_record_pending_confirm",
    "collecting_round",
    "offline_medical_record",
    "hos_sick_info",
    "preliminary_diagnosis",
    "patient_select_pending",
    "inquiry_progress",
    "prescription_reason",
]


class SessionService:
    """会话管理服务"""

    async def get_or_create(
            self,
            session_id: str,
            patient_id: str,
    ) -> dict[str, Any]:
        """获取已存在的会话，或创建新会话

        优先从 Redis 读取；未命中时从 MySQL 恢复；都没有则新建。
        """
        if await redis_client.exists(session_id):
            data = await redis_client.get_session_all(session_id)
            logger.info("恢复已有会话: %s (state=%s)", session_id, data.get("state"))
            return data

        # Redis 未命中，尝试从 MySQL 恢复
        mysql_session = await mysql_client.get_session(session_id)
        if mysql_session:
            logger.info("从 MySQL 恢复会话: %s", session_id)
            return await self._restore_from_mysql(mysql_session, session_id)

        # 都不存在，新建会话
        session_data = self._new_session_data(session_id, patient_id)

        # 写入 Redis
        for key, value in session_data.items():
            await redis_client.set_session_field(session_id, key, value)

        # 立即同步到 MySQL
        await self.sync_to_mysql(session_id)

        logger.info("新建会话: %s", session_id)
        return session_data

    def _new_session_data(self, session_id: str, patient_id: str) -> dict[str, Any]:
        """创建新会话的默认数据"""
        now = _utc_now_iso()
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "created_at": now,
            "updated_at": now,
            "state": SessionState.COLLECTING_BASIC.value,
            "paid": False,
            "patient_info": {},
            "patient_info_confirmed": {},
            "chief_complaint": "",
            "inquiry": {},
            "diagnosis": {},
            "prescription": {},
            "image_urls": [],
            "patient_mismatch": False,
            "mismatch_reason": "",
            "inquiry_progress": {},
        }

    async def _restore_from_mysql(
            self, mysql_session: ConsultationSession, session_id: str
    ) -> dict[str, Any]:
        """从 MySQL 恢复会话到 Redis"""
        session_data = {
            "session_id": mysql_session.session_id,
            "patient_id": mysql_session.patient_id,
            "state": mysql_session.status,
            "paid": mysql_session.paid,
            "patient_info": mysql_session.patient_info_collected or {},
            "patient_info_confirmed": mysql_session.patient_info_confirmed or {},
            "chief_complaint": mysql_session.chief_complaint or "",
            "inquiry": mysql_session.inquiry_json or {},
            "diagnosis": mysql_session.diagnosis_json or {},
            "prescription": mysql_session.prescription_json or {},
            "image_urls": mysql_session.image_urls or [],
            "patient_mismatch": mysql_session.patient_mismatch,
            "mismatch_reason": mysql_session.mismatch_reason or "",
        }
        if mysql_session.created_at:
            session_data["created_at"] = mysql_session.created_at.isoformat()
        if mysql_session.updated_at:
            session_data["updated_at"] = mysql_session.updated_at.isoformat()

        for key, value in session_data.items():
            await redis_client.set_session_field(session_id, key, value)

        # 同时恢复消息历史
        try:
            messages = await mysql_client.get_messages(session_id)
            if messages:
                msg_list = [
                    {"role": "user" if m.role == "patient" else "ai",
                     "content": m.content,
                     "images": m.images or []}
                    for m in messages
                ]
                await redis_client.set_session_field(session_id, "messages", msg_list)
                logger.info("已恢复 %d 条消息到 Redis", len(msg_list))
        except Exception as e:
            logger.warning("消息恢复失败（不影响会话）: %s", e)

        return session_data

    async def get_state(self, session_id: str) -> str | None:
        """获取当前会话状态"""
        return await redis_client.get_session_field(session_id, STATE_KEY)

    async def update_state(self, session_id: str, state: SessionState) -> None:
        """更新会话状态"""
        await redis_client.set_session_field(session_id, STATE_KEY, state.value)

    async def save_message(
            self,
            session_id: str,
            role: str,
            content: str,
            images: list[str] | None = None,
    ) -> None:
        """保存消息到 Redis（立即）+ MySQL（同步）

        角色命名转换：
          Redis: user / ai（与 _build_chat_messages 一致）
          MySQL: patient / assistant（符合表定义 ENUM）
        """
        # Redis: 追加到消息列表（保留 "user"/"ai" 格式）
        messages = await redis_client.get_session_field(session_id, "messages") or []
        messages.append({
            "role": role,
            "content": content,
            "images": images or [],
        })
        await redis_client.set_session_field(session_id, "messages", messages)

        # MySQL: 同步写入（角色名转换）
        try:
            from app.models.domain import ConsultationMessage

            mysql_role = "patient" if role == "user" else "assistant"
            msg = ConsultationMessage(
                session_id=session_id,
                role=mysql_role,
                content=content,
                images=images or [],
            )
            await mysql_client.save_message(msg)
        except Exception as e:
            logger.warning("消息写入 MySQL 失败（不影响对话）: %s", e)

    async def sync_to_mysql(self, session_id: str) -> None:
        """将会话状态同步到 MySQL"""
        try:
            data = await redis_client.get_session_all(session_id)
            if not data:
                return

            session_id = data.get("session_id", session_id)
            patient_id = data.get("patient_id", "")

            # 检查 MySQL 中是否有此会话
            existing = await mysql_client.get_session(session_id)

            if existing:
                await mysql_client.update_session(session_id, {
                    "status": data.get("state", ""),
                    "paid": data.get("paid", False),
                    "patient_info_collected": data.get("patient_info", {}),
                    "patient_info_confirmed": data.get("patient_info_confirmed", {}),
                    "chief_complaint": data.get("chief_complaint", ""),
                    "inquiry_json": data.get("inquiry", {}),
                    "diagnosis_json": data.get("diagnosis", {}),
                    "prescription_json": data.get("prescription", {}),
                    "image_urls": data.get("image_urls", []),
                    "patient_mismatch": data.get("patient_mismatch", False),
                    "mismatch_reason": data.get("mismatch_reason", ""),
                })
            else:
                session = ConsultationSession(
                    session_id=session_id,
                    patient_id=patient_id,
                    status=data.get("state", ""),
                    paid=data.get("paid", False),
                    patient_info_collected=data.get("patient_info", {}),
                    patient_info_confirmed=data.get("patient_info_confirmed", {}),
                    chief_complaint=data.get("chief_complaint", ""),
                    inquiry_json=data.get("inquiry", {}),
                    diagnosis_json=data.get("diagnosis", {}),
                    prescription_json=data.get("prescription", {}),
                    image_urls=data.get("image_urls", []),
                    patient_mismatch=data.get("patient_mismatch", False),
                    mismatch_reason=data.get("mismatch_reason", ""),
                )
                await mysql_client.create_session(session)

            logger.info("会话 %s 已同步到 MySQL", session_id)
        except Exception as e:
            logger.warning("MySQL 同步失败: %s", e)

    async def sync_session_state(
            self,
            session_id: str,
            final_state: dict[str, Any],
            deleted_fields: list[str] | None = None,
    ) -> None:
        """图成功后一次性写回会话状态（all-or-nothing，先删后写）

        LangGraph 重构后，节点不再直接写 Redis/MySQL，而是把更新放进 final state，
        由本方法按白名单统一写回：先删除 `deleted_fields`，再写白名单字段，
        最后同步 MySQL。图内异常时不会被调用，保证零写入。

        Args:
            session_id: 会话 ID
            final_state: 图的 final state（含全部会话字段）
            deleted_fields: 本轮到期的标记字段（如 med_record_pending_confirm），先删后跳过写回
        """
        deleted_fields = deleted_fields or []

        # 先删（到期标记）
        for field in deleted_fields:
            await redis_client.delete_session_field(session_id, field)

        # 再按白名单写回（被删除的字段不重写）
        for key in SESSION_FIELD_NAMES:
            if key in final_state and key not in deleted_fields:
                await redis_client.set_session_field(session_id, key, final_state[key])

        # 更新会话元数据（updated_at；created_at 建会话时写入）
        await redis_client.set_session_field(session_id, "updated_at", _utc_now_iso())

        # MySQL 同步（一次性；MySQL 侧 update 自动刷新 updated_at）
        await self.sync_to_mysql(session_id)

    async def set_field(self, session_id: str, field: str, value: Any) -> None:
        """设置会话的某个字段（直接写入 Redis）"""
        await redis_client.set_session_field(session_id, field, value)

    async def delete_field(self, session_id: str, field: str) -> None:
        """删除会话的某个字段"""
        await redis_client.delete_session_field(session_id, field)

    async def get_session_all(self, session_id: str) -> dict[str, Any]:
        """获取会话全部数据"""
        return await redis_client.get_session_all(session_id)
