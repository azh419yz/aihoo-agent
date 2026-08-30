"""问诊状态机 - 管理会话状态流转

状态流转:
  unpaid (paid=false):
    COLLECTING_BASIC → INQUIRY

  paid (paid=true):
    SELECTING_PATIENT → UPLOADING_IMAGES → DIAGNOSIS → PRESCRIBING
"""

from __future__ import annotations

from enum import Enum


class SessionState(str, Enum):
    """会话状态枚举"""

    # ---- 未付费阶段 ----
    COLLECTING_BASIC = "COLLECTING_BASIC"  # 收集基础信息（性别/年龄/过敏史/既往史）
    INQUIRY = "INQUIRY"  # 问诊中（持续收集主诉）

    # ---- 已付费阶段 ----
    PRELIMINARY_DIAGNOSIS = "PRELIMINARY_DIAGNOSIS"  # 支付后初步辨证（双库检索+初步建议）
    SELECTING_PATIENT = "SELECTING_PATIENT"  # 选择就诊人
    UPLOADING_IMAGES = "UPLOADING_IMAGES"  # 上传舌照/面照
    DIAGNOSIS = "DIAGNOSIS"  # 辨病辨证
    PRESCRIBING = "PRESCRIBING"  # 开具处方

    # ---- 异常 ----
    ERROR = "ERROR"  # 错误状态

class ActionType(str, Enum):
    """业务动作标识"""

    CHAT = "CHAT"  # 普通对话 / 持续收集信息
    COLLECT_BASIC_INFO = "COLLECT_BASIC_INFO"  # 收集基础信息
    SELECT_PATIENT = "SELECT_PATIENT"  # 选择就诊人
    UPLOAD_IMAGES = "UPLOAD_IMAGES"  # 上传舌照/面照
    DIAGNOSIS = "DIAGNOSIS"  # 辨病辨证
    PRESCRIBE = "PRESCRIBE"  # 开具处方


# 各状态下支持的 Action
STATE_ACTIONS: dict[SessionState, list[ActionType]] = {
    SessionState.COLLECTING_BASIC: [ActionType.CHAT, ActionType.COLLECT_BASIC_INFO],
    SessionState.INQUIRY: [ActionType.CHAT],
    SessionState.PRELIMINARY_DIAGNOSIS: [ActionType.CHAT, ActionType.DIAGNOSIS],
    SessionState.SELECTING_PATIENT: [ActionType.SELECT_PATIENT, ActionType.CHAT],
    SessionState.UPLOADING_IMAGES: [ActionType.CHAT, ActionType.UPLOAD_IMAGES],
    SessionState.DIAGNOSIS: [ActionType.CHAT, ActionType.DIAGNOSIS, ActionType.PRESCRIBE],
    SessionState.PRESCRIBING: [ActionType.CHAT, ActionType.PRESCRIBE],
}


# 各状态激活的知识库
STATE_KNOWLEDGE_BASE: dict[SessionState, str] = {
    SessionState.COLLECTING_BASIC: "general",
    SessionState.INQUIRY: "general",
    SessionState.PRELIMINARY_DIAGNOSIS: "both",
    SessionState.SELECTING_PATIENT: "both",
    SessionState.UPLOADING_IMAGES: "both",
    SessionState.DIAGNOSIS: "both",
    SessionState.PRESCRIBING: "expert",
}


# 各状态下主诉是否持续收集
STATE_CHIEF_COMPLAINT_ACTIVE: dict[SessionState, bool] = {
    SessionState.COLLECTING_BASIC: False,
    SessionState.INQUIRY: True,
    SessionState.PRELIMINARY_DIAGNOSIS: False,
    SessionState.SELECTING_PATIENT: False,
    SessionState.UPLOADING_IMAGES: True,
    SessionState.DIAGNOSIS: True,
    SessionState.PRESCRIBING: False,
}
