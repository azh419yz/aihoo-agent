"""Agent 对话 API 接口（Controller 层）

薄 controller 层：参数校验 → 调用 Service → 返回统一响应。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.common.base_response import BaseResponse
from app.common.dependencies import get_consultation_service, get_session_service
from app.common.exceptions import SessionNotFoundError
from app.models.chat_schema import (
    ChatRequest,
    CreateSessionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


@router.post("/chat", response_model=BaseResponse)
async def chat(request: ChatRequest):
    """Agent 对话接口

    处理患者的问诊对话，返回 Agent 回复和业务动作。
    """
    service = get_consultation_service()
    result = await service.process_chat(request)
    return BaseResponse.success(data=result)


@router.post("/session", response_model=BaseResponse)
async def create_session(request: CreateSessionRequest):
    """新建问诊会话"""
    session_service = get_session_service()
    data = await session_service.get_or_create(
        request.session_id, request.patient_id
    )

    # 同步到 MySQL
    await session_service.sync_to_mysql(request.session_id)

    return BaseResponse.success(data={
        "session_id": request.session_id,
        "patient_id": request.patient_id,
        "state": data.get("state", "COLLECTING_BASIC"),
    })


@router.get("/session/{session_id}", response_model=BaseResponse)
async def get_session(session_id: str):
    """获取会话状态"""
    session_service = get_session_service()
    data = await session_service.get_session_all(session_id)

    if not data:
        raise SessionNotFoundError(session_id)

    return BaseResponse.success(data={
        "session_id": session_id,
        "state": data.get("state"),
        "paid": data.get("paid", False),
        "chief_complaint": data.get("chief_complaint", ""),
        "patient_mismatch": data.get("patient_mismatch", False),
        "patient_info": data.get("patient_info", {}),
    })


@router.get("/session/{session_id}/messages", response_model=BaseResponse)
async def get_messages(session_id: str, limit: int = 50):
    """获取历史消息"""
    from app.storage.mysql import mysql_client

    messages = await mysql_client.get_messages(session_id, limit=limit)
    return BaseResponse.success(data=[
        {
            "role": msg.role,
            "content": msg.content,
            "images": msg.images or [],
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in messages
    ])
