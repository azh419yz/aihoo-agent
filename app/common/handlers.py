"""全局异常处理器"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.common.base_response import BaseResponse
from app.common.exceptions import AppException
from app.common.response_codes import ResponseCode

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器"""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning("业务异常: [%d] %s", exc.code, exc.message)
        return JSONResponse(
            status_code=200,
            content=BaseResponse.error(code=exc.code, message=exc.message, data=exc.data).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("参数校验失败: %s", exc.errors())
        return JSONResponse(
            status_code=200,
            content=BaseResponse.error(
                code=ResponseCode.BAD_REQUEST.code,
                message=str(exc.errors()[0]["msg"]) if exc.errors() else "请求参数校验失败",
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("值错误: %s", exc)
        return JSONResponse(
            status_code=200,
            content=BaseResponse.error(
                code=ResponseCode.BAD_REQUEST.code,
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=200,
            content=BaseResponse.error(
                code=ResponseCode.INTERNAL_ERROR.code,
                message="服务器内部错误",
            ).model_dump(),
        )
