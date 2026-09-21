"""会话管理服务"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.agent.state_machine import SessionState
from app.common.exceptions import SessionNotFoundError
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

# 会话字段的期望类型（Redis 往返的类型归一化用）。
#
# 背景：Redis hash 只能存字符串 —— 写侧对 dict/list/bool 做 json.dumps、str 原样写入；
# 读侧 `get_session_all` 对每个字段 json.loads，于是「看起来像 JSON 字面量」的字符串
# 会被还原成别的类型（"5" → 5、"true" → True、"null" → None）。而 Pydantic v2 不再做
# int → str 的隐式转换，`ConsultationSession(patient_id=5)` 会直接校验失败 ——
# 线上曾因此三周 490 次 MySQL 同步失败，会话状态从未落库（2026-09-20 工单 IKGUP0）。
#
# 这里按字段声明类型把值还原回原类型。新增会话字段时请一并登记（测试会校验覆盖率）。
SESSION_FIELD_TYPES: dict[str, type] = {
    # 会话元数据：不在 SESSION_FIELD_NAMES 写回白名单内，但同样存在 Redis 里。
    # patient_id 是线上事故的主角（Java 侧传数字型 ID "5"），必须还原成 str。
    "session_id": str,
    "patient_id": str,
    "created_at": str,
    "updated_at": str,
    # 会话业务字段（与 SESSION_FIELD_NAMES 一一对应）
    "state": str,
    "paid": bool,
    "patient_info": dict,
    "patient_info_confirmed": dict,
    "chief_complaint": str,
    "inquiry": dict,
    "diagnosis": dict,
    "prescription": dict,
    "prescription_reason": dict,
    "image_urls": list,
    "tongue_analysis": list,
    "face_analysis": list,
    "patient_mismatch": bool,
    "mismatch_reason": str,
    "med_record_pending_confirm": bool,
    "collecting_round": int,
    "offline_medical_record": dict,
    "hos_sick_info": dict,
    "preliminary_diagnosis": dict,
    "patient_select_pending": bool,
    "inquiry_progress": dict,
}

# 允许「会话不存在时直接新建」的请求动作（真正的冷启动）。
# 其余动作（SELECT_PATIENT / UPLOAD_IMAGES / DIAGNOSIS / PRESCRIBE）只可能发生在
# 已存在会话的中途，若此时 Agent 侧查无此会话，说明状态已丢失，不能静默新建。
COLD_START_ACTIONS: frozenset[str] = frozenset({"CHAT", "COLLECT_BASIC_INFO"})


def _restore_field_types(data: dict[str, Any]) -> dict[str, Any]:
    """把 Redis 往返后类型走样的字段还原回声明类型（就地修改并返回）。

    当前只有 str 字段需要还原：dict/list/bool/int 经 json.dumps → json.loads
    往返后类型是保真的，唯独字符串在缺少类型信息时会被「猜」成 int/bool/None。
    """
    for field, expected in SESSION_FIELD_TYPES.items():
        if expected is not str or field not in data:
            continue
        value = data[field]
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, (dict, list)):
            logger.warning(
                "会话字段 %s 期望 str，实为 %s，保持原值", field, type(value).__name__
            )
            continue
        data[field] = str(value)
    return data


# Redis 会话字段 → MySQL 列名（仅列不同名的；其余同名）。
# `sync_to_mysql` 的 create / update 两条分支共用同一映射，避免两边字段集不一致
# （历史上 create 只写 11 列、update 也只写 11 列，MySQL 兜底恢复因此长期残缺）。
_REDIS_TO_MYSQL_FIELD: dict[str, str] = {
    "state": "status",
    "patient_info": "patient_info_collected",
    "inquiry": "inquiry_json",
    "diagnosis": "diagnosis_json",
    "prescription": "prescription_json",
}


def _to_mysql_updates(data: dict[str, Any]) -> dict[str, Any]:
    """把 Redis 会话数据映射成 MySQL 列 → 值（覆盖 SESSION_FIELD_NAMES 全字段）"""
    updates: dict[str, Any] = {}
    for field in SESSION_FIELD_NAMES:
        if field not in data:
            continue
        value = data[field]
        if value is None:
            continue
        updates[_REDIS_TO_MYSQL_FIELD.get(field, field)] = value
    return updates


class SessionService:
    """会话管理服务"""

    async def get_or_create(
            self,
            session_id: str,
            patient_id: str,
            action: str | None = None,
            paid: bool | None = None,
    ) -> dict[str, Any]:
        """获取已存在的会话，或创建新会话

        优先从 Redis 读取；未命中时从 MySQL 恢复；都没有则新建。

        Args:
            session_id: 会话 ID
            patient_id: 患者 ID
            action: 本次请求动作。传入后用于识别「状态丢失」——若该会话本不该是新的
                （见 `_is_state_lost`），则拒绝静默新建并抛 SessionNotFoundError。
                显式建会话接口（POST /session）不传，行为保持为直接新建。
            paid: 本次请求的付费标记，用于 `_is_state_lost` 判断。
        """
        if await redis_client.exists(session_id):
            data = _restore_field_types(await redis_client.get_session_all(session_id))
            logger.info("恢复已有会话: %s (state=%s)", session_id, data.get("state"))
            return data

        # Redis 未命中，尝试从 MySQL 恢复
        mysql_session = await mysql_client.get_session(session_id)
        if mysql_session:
            logger.info("从 MySQL 恢复会话: %s", session_id)
            return await self._restore_from_mysql(mysql_session, session_id)

        # 两边都没有：若请求上下文表明该会话本该存在，说明状态已丢失。
        # 此时新建只会把「中断」伪装成「新会话」，让下游拿到错误状态去执行
        # （2026-09-20 工单 IKGUP0：付费会话被重置为 COLLECTING_BASIC → PRESCRIBE 报 2002）
        if self._is_state_lost(action, paid):
            logger.error(
                "会话状态丢失，拒绝静默新建: session=%s patient=%s action=%s paid=%s "
                "（Redis 与 MySQL 均无此会话，请优先检查 MySQL 同步是否正常）",
                session_id, patient_id, action, paid,
            )
            raise SessionNotFoundError(session_id)

        # 都不存在，新建会话
        session_data = self._new_session_data(session_id, patient_id)

        # 写入 Redis
        for key, value in session_data.items():
            await redis_client.set_session_field(session_id, key, value)

        # 立即同步到 MySQL
        await self.sync_to_mysql(session_id)

        logger.info("新建会话: %s", session_id)
        return session_data

    @staticmethod
    def _is_state_lost(action: str | None, paid: bool | None) -> bool:
        """判断「查无此会话」是否意味着状态丢失，而不是正常的首次冷启动

        - action 未传入（显式建会话接口）→ 否，允许新建
        - 已付费的请求 → 会话必然已存在过，查无此会话只能是状态丢失
        - action 属于付费后专属动作（SELECT_PATIENT/UPLOAD_IMAGES/DIAGNOSIS/PRESCRIBE）
          → 同上，这些动作不可能出现在一轮全新的会话里
        """
        if action is None:
            return False
        if paid:
            return True
        return action not in COLD_START_ACTIONS

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
        """从 MySQL 恢复会话到 Redis（字段集与 SESSION_FIELD_NAMES 对齐）"""
        session_data: dict[str, Any] = {
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
            "prescription_reason": mysql_session.prescription_reason or {},
            "image_urls": mysql_session.image_urls or [],
            "patient_mismatch": mysql_session.patient_mismatch,
            "mismatch_reason": mysql_session.mismatch_reason or "",
            # 编排中间态（2026-09-21 补）：这几个字段缺失会让恢复出来的会话
            # 「看着在正确的状态、实际没有进度」—— 系统问诊重头问、就诊人重选
            "inquiry_progress": mysql_session.inquiry_progress or {},
            "preliminary_diagnosis": mysql_session.preliminary_diagnosis or {},
            "hos_sick_info": mysql_session.hos_sick_info or {},
            "tongue_analysis": mysql_session.tongue_analysis or [],
            "face_analysis": mysql_session.face_analysis or [],
            "collecting_round": mysql_session.collecting_round or 0,
            "med_record_pending_confirm": mysql_session.med_record_pending_confirm,
            "offline_medical_record": mysql_session.offline_medical_record or {},
            "patient_select_pending": mysql_session.patient_select_pending,
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
            # 不中断对话，但升级为 ERROR：消息审计同样曾长期静默失败
            logger.error(
                "消息写入 MySQL 失败（对话继续，但审计缺失）: session=%s err=%s",
                session_id, e,
            )

    async def sync_to_mysql(self, session_id: str) -> None:
        """将会话状态全字段同步到 MySQL（字段集由 SESSION_FIELD_NAMES 决定）"""
        try:
            data = _restore_field_types(await redis_client.get_session_all(session_id))
            if not data:
                return

            session_id = data.get("session_id", session_id)
            updates = _to_mysql_updates(data)

            # 检查 MySQL 中是否有此会话
            existing = await mysql_client.get_session(session_id)

            if existing:
                await mysql_client.update_session(session_id, updates)
            else:
                session = ConsultationSession(
                    session_id=session_id,
                    # 显式 str：即使类型归一化被绕过，也不让 patient_id 变成 int
                    patient_id=str(data.get("patient_id") or ""),
                    **updates,
                )
                await mysql_client.create_session(session)

            logger.info("会话 %s 已同步到 MySQL", session_id)
        except Exception as e:
            # 不中断对话，但必须留 ERROR 级记录：历史上这里只打 WARNING，
            # 导致连续三周 490 次同步失败无人察觉（2026-09-20 工单 IKGUP0）。
            logger.error(
                "MySQL 同步失败（会话状态未落库）: session=%s err=%s", session_id, e
            )

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
        """获取会话全部数据（类型归一化后）"""
        return _restore_field_types(await redis_client.get_session_all(session_id))
