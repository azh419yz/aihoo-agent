"""问诊编排服务 - 核心业务逻辑

连接 Controller -> Agent/Storage 各层，用 LangGraph 状态图驱动流程。

状态处理逻辑已迁移到 app/agent/graph.py 的节点中；本服务只负责：
  session 装载 → action 校验 → 图执行 → 一次性持久化 → 保存消息 → 组装响应。
"""

from __future__ import annotations

import logging

from app.agent.graph import build_consultation_graph
from app.agent.orchestrator import LLMOrchestrator
from app.agent.state_machine import PAID_STATES, STATE_ACTIONS, ActionType, SessionState
from app.common.exceptions import (
    LLMServiceError,
    PaymentRequiredError,
    StateTransitionError,
)
from app.models.chat_schema import ChatRequest, ChatResponse
from app.service.session_service import SESSION_FIELD_NAMES, SessionService

logger = logging.getLogger(__name__)


class ConsultationService:
    """问诊编排服务"""

    def __init__(self):
        self.orchestrator = LLMOrchestrator()
        self.session_service = SessionService()
        # compiled graph 跨请求复用安全（仅 ainvoke，非 streaming）
        self.graph = build_consultation_graph(self.orchestrator)

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """处理对话请求的主流程

        1. 加载会话 → 2. action 校验 → 3. 构造 initial state
        → 4. 图执行 → 5. 一次性持久化 → 6. 保存消息 → 7. 返回响应
        """
        session_id = request.session_id

        # 1. 加载/创建会话（传入 action/paid 用于识别状态丢失，不接受静默新建）
        session_data = await self.session_service.get_or_create(
            session_id, request.patient_id,
            action=request.action, paid=request.paid,
        )

        # 2. 前置 action 校验（原 StateMachine.get_allowed_actions，类已退役删除）
        state_value = session_data.get("state", SessionState.COLLECTING_BASIC.value)
        current_state = SessionState(state_value)
        action = ActionType(request.action)
        allowed = STATE_ACTIONS.get(current_state, [ActionType.CHAT])
        if action not in allowed:
            raise StateTransitionError(
                current_state=state_value,
                target_state=f"action={action.value}",
            )

        # 3. 前置付费校验：付费后状态必须显式带 paid=true（2026-09-21 加）
        # 判据用「请求的 paid」而非「会话 paid」：会话 paid 只在 INQUIRY 阶段被
        # 消费一次并写入，此后付费后节点内部一律硬编码 paid=True，导致请求传
        # false 也能静默推进（线上实测 UPLOADING_IMAGES + paid=false 被放行）。
        # 按调用契约（api-调用手册 §3.3/§3.5），付费后每轮都应带 paid=true。
        if current_state in PAID_STATES and not request.paid:
            logger.error(
                "付费后状态缺少付费标记，拒绝处理: session=%s state=%s paid=%s",
                session_id, state_value, request.paid,
            )
            raise PaymentRequiredError(current_state=state_value)

        # 4. 构造 initial state（会话持久化字段 + 本次请求输入）
        initial: dict = {
            key: session_data[key]
            for key in SESSION_FIELD_NAMES
            if key in session_data
        }
        if session_data.get("messages") is not None:
            initial["messages"] = session_data["messages"]
        initial["_deleted_fields"] = []
        initial.update({
            "request_message": request.message,
            "request_action": request.action,
            "request_paid": request.paid,
            "request_hos_sick_info": (
                request.hos_sick_info.model_dump() if request.hos_sick_info else None
            ),
            "request_medical_record_urls": request.medical_record_urls or [],
            "request_tongue_urls": request.tongue_urls or [],
            "request_face_urls": request.face_urls or [],
        })

        # 4. 图执行（LLM 配置错误 → LLMServiceError(3000)，同现状）
        try:
            final = await self.graph.ainvoke(initial)
        except ValueError as e:
            # LLM 配置错误（如未配置 API Key）
            raise LLMServiceError(
                code=3000,
                message=str(e) or "LLM 服务调用失败，请检查 API Key 配置",
            )

        # 5. 图成功后一次性持久化（all-or-nothing，先删后写）
        await self.session_service.sync_session_state(
            session_id, final, final.get("_deleted_fields") or []
        )

        # 6. 保存消息（Redis 立即）
        all_images = []
        for urls in (request.medical_record_urls, request.tongue_urls, request.face_urls):
            if urls:
                all_images.extend(urls)
        await self.session_service.save_message(
            session_id=session_id,
            role="user",
            content=request.message,
            images=all_images or None,
        )
        await self.session_service.save_message(
            session_id=session_id,
            role="ai",
            content=final["response_text"],
        )

        # 7. 返回响应
        return ChatResponse(
            session_id=session_id,
            message=final["response_text"],
            action=final["response_action"].value,
            meta=final["response_data"],
        )
