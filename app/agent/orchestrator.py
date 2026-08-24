"""轻量 LLM 编排器（OpenAI-compatible 统一调用）

统一走 qwen3.8-max 一个多模态模型，文本对话 / 图片分析 / 结构化输出
全部通过 openai AsyncOpenAI client 调百炼 compatible-mode 网关（企业 MaaS）。

无 LangChain、无 dashscope 依赖；每一步由调用方明确编排。
全链路关闭思考模式（enable_thinking=false），保证快速直接、结构化输出稳定。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agent.prompts import (
    build_patient_confirm_prompt,
    build_patient_match_prompt,
    build_patient_match_system_message,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """轻量 LLM 编排器

    提供 chat() / chat_with_vl() / ainvoke_structured() 三个核心方法，
    外加 match_patient_info() / confirm_patient_info() 两个业务封装。
    """

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    # ============================================================
    # 客户端懒加载（避免无 API Key 时构造就报错）
    # ============================================================

    @property
    def client(self) -> AsyncOpenAI:
        """OpenAI-compatible 异步客户端（懒加载，走企业网关）"""
        if self._client is None:
            if not settings.DASHSCOPE_API_KEY:
                raise ValueError(
                    "DASHSCOPE_API_KEY 未配置，请在 .env 文件中设置"
                )
            self._client = AsyncOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.DASHSCOPE_BASE_URL,
            )
        return self._client

    # ============================================================
    # 核心方法
    # ============================================================

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """直接 LLM 对话（文本或图片消息均可，OpenAI dict 格式）

        Args:
            messages: 消息列表（role: system/user/assistant；content 为 str 或 list）
            temperature: 可选，覆盖默认温度（默认 0.7）

        Returns:
            LLM 回复文本
        """
        kwargs: dict[str, Any] = (
            {"temperature": temperature} if temperature is not None else {"temperature": 0.7}
        )
        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            extra_body={"enable_thinking": False},
            **kwargs,
        )
        return response.choices[0].message.content

    async def chat_with_vl(
        self,
        messages: list[dict],
    ) -> str:
        """多模态 LLM 对话（分析舌照/面照/病历等图片）

        与 chat() 共用同一个 qwen3.8-max 模型，仅用低温度保证分析确定性。

        Args:
            messages: 消息列表，图片用 {"type":"image_url","image_url":{"url":...}}

        Returns:
            LLM 回复文本
        """
        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    async def ainvoke_structured(
        self,
        schema: type[BaseModel],
        messages: list[dict],
        *,
        max_retries: int = 2,
    ) -> Any:
        """结构化输出（json_schema 严格模式 + 重试 + None 兜底）

        用 response_format=json_schema 强制模型输出符合 Pydantic schema 的 JSON。
        qwen3.8-max 偶发返回空或非法 JSON，统一在此做补偿重试，
        最终仍失败时返回 None，由调用方兜底（不再冒泡成 500）。

        Args:
            schema: Pydantic BaseModel 子类
            messages: OpenAI 消息列表
            max_retries: 失败后的重试次数（默认 2）

        Returns:
            解析后的 Pydantic 模型；最终失败返回 None
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": schema.model_json_schema(),
                        },
                    },
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content
                if content:
                    return schema.model_validate(json.loads(content))
                logger.warning(
                    "结构化输出返回空（第 %d/%d 次）: %s",
                    attempt + 1, max_retries + 1, schema.__name__,
                )
            except Exception as e:
                last_exc = e
                logger.warning(
                    "结构化输出调用异常（第 %d/%d 次）: %s",
                    attempt + 1, max_retries + 1, e,
                )
        if last_exc:
            logger.error("结构化输出最终失败 %s: %s", schema.__name__, last_exc)
        else:
            logger.error("结构化输出最终为空 %s", schema.__name__)
        return None

    async def match_patient_info(
        self,
        collected: dict[str, Any],
        confirmed: dict[str, Any],
    ) -> tuple[bool, str]:
        """校验患者信息是否匹配

        Args:
            collected: Agent 收集的患者信息
            confirmed: 后端选择的就诊人信息

        Returns:
            (is_match, reason)
        """
        from app.agent.structured_output import PatientMatchResult

        prompt = build_patient_match_prompt(collected, confirmed)

        result = await self.ainvoke_structured(
            PatientMatchResult,
            [
                {"role": "system", "content": build_patient_match_system_message()},
                {"role": "system", "content": prompt},
            ],
        )
        if result is None:
            logger.error("就诊人匹配结构化输出为空，按不匹配保守处理")
            return False, "就诊人信息校验失败，请重新确认"
        return result.is_match, result.reason

    async def confirm_patient_info(
        self,
        collected: dict[str, Any],
        confirmed: dict[str, Any],
        mismatch_reason: str,
        user_message: str,
    ) -> tuple[str, str]:
        """语义分析用户对就诊人确认的响应

        当系统检测到就诊人信息不匹配并展示给用户后，
        用 LLM 语义理解判断用户真实意图。

        Args:
            collected: 对话收集的患者信息
            confirmed: 系统记录的就诊人信息
            mismatch_reason: 已检测到的不匹配原因
            user_message: 用户最新回复

        Returns:
            (action, analysis)
            action: "confirm" | "disagree" | "unknown"
        """
        from app.agent.structured_output import PatientConfirmResult

        prompt = build_patient_confirm_prompt(collected, confirmed, mismatch_reason)

        result = await self.ainvoke_structured(
            PatientConfirmResult,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
        )
        if result is None:
            logger.error("就诊人确认语义分析输出为空，按不明确处理")
            return "unknown", ""
        return result.action.value, result.analysis
