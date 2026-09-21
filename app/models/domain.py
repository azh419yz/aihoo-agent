"""领域模型 - 数据库实体映射"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConsultationSession(BaseModel):
    """问诊会话 - 对应 consultation_sessions 表"""

    id: int | None = Field(default=None, description="自增主键（无业务意义）")
    session_id: str = Field(..., description="会话 UUID（业务标识）")
    patient_id: str = Field(..., description="患者 ID (UUID)")
    status: str = Field(default="COLLECTING_BASIC", description="当前状态")
    paid: bool = Field(default=False, description="是否已付费")

    # Agent 收集的患者信息
    patient_info_collected: dict[str, Any] | None = Field(default=None, description="Agent 对话收集的信息")
    # 后端传入的就诊人信息
    patient_info_confirmed: dict[str, Any] | None = Field(default=None, description="后端选择的就诊人信息")

    chief_complaint: str | None = Field(default=None, description="主诉")
    inquiry_json: dict[str, Any] | None = Field(default=None, description="问诊信息")
    diagnosis_json: dict[str, Any] | None = Field(default=None, description="辨病辨证结果")
    prescription_json: dict[str, Any] | None = Field(default=None, description="处方信息")
    prescription_reason: dict[str, Any] | None = Field(
        default=None, description="开方选案分析原因（Redis 审计用，不展示给用户）"
    )
    image_urls: list[str] | None = Field(default=None, description="舌照/面照 URL 列表")

    # 问诊编排中间态（2026-09-21 补列：此前这些字段只在 Redis，
    # Redis 键过期后从 MySQL 恢复会整段丢失，用户被迫重复问诊）
    inquiry_progress: dict[str, Any] | None = Field(
        default=None, description="问诊维度进度（主诉 + 6 个系统维度 → bool）"
    )
    preliminary_diagnosis: dict[str, Any] | None = Field(
        default=None, description="付费后初步辨证结果"
    )
    hos_sick_info: dict[str, Any] | None = Field(
        default=None, description="后端传入的就诊人信息（原始入参）"
    )
    tongue_analysis: list[Any] | None = Field(default=None, description="舌象分析结果")
    face_analysis: list[Any] | None = Field(default=None, description="面象分析结果")
    collecting_round: int | None = Field(default=None, description="基础信息收集轮次")
    med_record_pending_confirm: bool = Field(default=False, description="病历待用户确认")
    offline_medical_record: dict[str, Any] | None = Field(
        default=None, description="线下病历处理状态（已处理图片 URL 等）"
    )
    patient_select_pending: bool = Field(default=False, description="就诊人信息待用户确认")

    patient_mismatch: bool = Field(default=False, description="患者信息是否不匹配")
    mismatch_reason: str | None = Field(default=None, description="不匹配原因")

    created_at: datetime | None = Field(default=None, description="创建时间")
    updated_at: datetime | None = Field(default=None, description="更新时间")


class ConsultationMessage(BaseModel):
    """对话记录 - 对应 consultation_messages 表"""

    id: int | None = Field(default=None, description="自增主键")
    session_id: str = Field(..., description="会话 UUID")
    role: str = Field(..., description="角色: patient / assistant")
    content: str = Field(..., description="消息内容")
    images: list[str] | None = Field(default=None, description="图片 URL 列表")
    created_at: datetime | None = Field(default=None, description="创建时间")
