"""Agent 工具函数

提示：这些函数不再使用 @tool 装饰器或 create_agent()。
它们由 Service 层代码直接调用，不是 LLM 调用的 function calling。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.orchestrator import LLMOrchestrator
from app.agent.prompts import build_prescription_prompt, build_prescription_system_message
from app.agent.structured_output import PrescriptionResult
from app.knowledge.qianfan import retrieve as kb_retrieve, format_retrieval_context
from app.knowledge.tcm_matcher import tcm_matcher

logger = logging.getLogger(__name__)


async def retrieve_knowledge(
        query: str,
        knowledge_base: str = "general",
        top_k: int = 5,
        diseases: list[str] | None = None,
        syndromes: list[str] | None = None,
        gender: str | None = None,
        age: int | None = None,
) -> str:
    """检索中医知识库获取专业信息

    自动用 tcm_matcher 匹配疾病/证型来增强查询词（如果未提供上下文）。
    通用/专家双库使用不同检索策略，支持附加性别/年龄提高命中率。

    Args:
        query: 检索问题（患者症状描述）
        knowledge_base: 知识库类型 (general/expert/both)
        top_k: 返回结果数量
        diseases: 可选，已匹配的疾病名称（提供后跳过 tcm_matcher 匹配）
        syndromes: 可选，已匹配的证型名称
        gender: 可选，患者性别，用于增强查询
        age: 可选，患者年龄，用于增强查询

    Returns:
        格式化后的知识文本
    """
    try:
        # 如果未提供疾病/证型上下文，自动用 tcm_matcher 匹配
        matcher_diseases = diseases
        matcher_syndromes = syndromes

        if not matcher_diseases and not matcher_syndromes and tcm_matcher.loaded:
            matched_diseases = tcm_matcher.match_diseases(query, top_k=3)
            matched_syndromes = tcm_matcher.match_syndromes(query, top_k=3)
            matcher_diseases = [d["name"] for d in matched_diseases if d.get("name")]
            matcher_syndromes = [s["name"] for s in matched_syndromes if s.get("name")]

        results = await kb_retrieve(
            query=query,
            knowledge_base=knowledge_base,
            top_k=top_k,
            diseases=matcher_diseases,
            syndromes=matcher_syndromes,
            gender=gender,
            age=age,
        )
        return format_retrieval_context(results)
    except Exception as e:
        logger.warning("知识库检索失败: %s", e)
        return "[知识库暂时不可用]"


async def generate_prescription(
        diagnosis: dict[str, str],
        chief_complaint: str | None = None,
        patient_info: dict[str, Any] | None = None,
        knowledge_context: str | None = None,
        base_formula: str | None = None,
        inquiry_info: dict[str, Any] | None = None,
        orchestrator: LLMOrchestrator | None = None,
) -> dict[str, Any]:
    """根据辨证结果生成处方

    参考来源：推荐主方（tcm_syndrome）→ 历史处方案例（expert 知识库检索，
    经 性别+年龄±10 过滤，已含在 knowledge_context）→ 知识库参考。

    Args:
        diagnosis: 辨证结果（包含 disease, syndrome 等）
        chief_complaint: 主诉
        patient_info: 患者信息
        knowledge_context: 知识库检索上下文（含 expert 库相似病例）
        base_formula: 推荐主方名称（如"右归丸加减"），来自 tcm_syndrome.recommended_formula
        orchestrator: LLM 编排器实例

    Returns:
        处方信息（新格式：drugList + instruction）
    """
    if orchestrator is None:
        orchestrator = LLMOrchestrator()

    prompt = build_prescription_prompt(
        diagnosis=diagnosis,
        chief_complaint=chief_complaint,
        patient_info=patient_info,
        knowledge_context=knowledge_context,
        base_formula=base_formula,
        inquiry_info=inquiry_info,
    )

    result = await orchestrator.ainvoke_structured(
        PrescriptionResult,
        [
            {"role": "system", "content": build_prescription_system_message()},
            {"role": "system", "content": prompt},
        ],
    )
    if result is None:
        logger.error(
            "处方生成失败（结构化输出为空）: %s / %s",
            diagnosis.get("disease"), diagnosis.get("syndrome"),
        )
        return {
            "disease": diagnosis.get("disease", ""),
            "syndrome": diagnosis.get("syndrome", ""),
            "drugList": [],
            "instruction": {},
        }

    return result.model_dump()
