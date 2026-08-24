"""自定义业务异常"""

from __future__ import annotations

from app.common.response_codes import ResponseCode


class AppException(Exception):
    """应用异常基类"""

    def __init__(self, code: int, message: str, data: object = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

    @classmethod
    def from_code(cls, resp_code: ResponseCode, message: str | None = None, data: object = None) -> "AppException":
        return cls(code=resp_code.code, message=message or resp_code.message, data=data)


class BusinessException(AppException):
    """业务逻辑异常"""
    pass


class SessionNotFoundError(BusinessException):
    """会话不存在"""

    def __init__(self, session_id: str):
        super().__init__(
            code=ResponseCode.SESSION_NOT_FOUND.code,
            message=f"会话不存在: {session_id}",
        )


class SessionExpiredError(BusinessException):
    """会话已过期"""
    pass


class StateTransitionError(BusinessException):
    """状态转移非法"""

    def __init__(self, current_state: str, target_state: str):
        super().__init__(
            code=ResponseCode.INVALID_STATE_TRANSITION.code,
            message=f"非法状态转移: {current_state} → {target_state}",
            data={"current": current_state, "target": target_state},
        )


class LLMServiceError(AppException):
    """LLM 服务异常"""
    pass


class KnowledgeBaseError(AppException):
    """知识库检索异常"""
    pass


class StorageError(AppException):
    """存储层异常"""
    pass
