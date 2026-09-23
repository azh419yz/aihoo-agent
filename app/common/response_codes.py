"""业务响应码定义"""

from __future__ import annotations

from enum import Enum


class ResponseCode(Enum):
    """业务响应码（整数 code + 文字 message）"""

    # 成功
    SUCCESS = (0, "success")

    # 通用错误 (1xxx)
    BAD_REQUEST = (1000, "请求参数错误")
    UNAUTHORIZED = (1001, "未授权")
    FORBIDDEN = (1002, "无权限")
    NOT_FOUND = (1003, "资源不存在")
    INTERNAL_ERROR = (1004, "服务器内部错误")
    SERVICE_UNAVAILABLE = (1005, "服务暂不可用")

    # 业务错误 (2xxx)
    SESSION_NOT_FOUND = (2000, "会话不存在")
    SESSION_EXPIRED = (2001, "会话已过期")
    INVALID_STATE_TRANSITION = (2002, "状态流转非法")
    PATIENT_MISMATCH = (2003, "就诊人信息不匹配")
    DUPLICATE_SESSION = (2004, "会话已存在")
    PAYMENT_REQUIRED = (2005, "需付费后使用")

    # 三方服务错误 (3xxx)
    LLM_SERVICE_ERROR = (3000, "LLM 服务调用失败")
    KNOWLEDGE_BASE_ERROR = (3001, "知识库检索失败")
    REDIS_ERROR = (3100, "Redis 操作失败")
    MYSQL_ERROR = (3101, "MySQL 操作失败")

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
