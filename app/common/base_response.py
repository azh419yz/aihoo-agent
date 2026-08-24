"""统一响应模型"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.common.response_codes import ResponseCode

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """统一 API 响应包装"""

    code: int = ResponseCode.SUCCESS.code
    message: str = ResponseCode.SUCCESS.message
    data: T | None = None
    request_id: str | None = None

    @classmethod
    def success(cls, data: T | None = None, message: str = "success") -> "BaseResponse[T]":
        return cls(code=ResponseCode.SUCCESS.code, message=message, data=data)

    @classmethod
    def error(cls, code: int, message: str, data: Any = None) -> "BaseResponse":
        return cls(code=code, message=message, data=data)
