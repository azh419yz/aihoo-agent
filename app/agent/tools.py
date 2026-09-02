"""Agent 工具函数

提示：这些函数不再使用 @tool 装饰器或 create_agent()。
它们由 Service 层代码直接调用，不是 LLM 调用的 function calling。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.orchestrator import LLMOrchestrator
from app.agent.prompts import (
    build_case_selection_prompt,
    build_prescription_instruction_prompt,
)
from app.agent.structured_output import CaseSelectionResult, InstructionInfo
from app.knowledge.qianfan import (
    format_retrieval_context,
    parse_case_prescription,
    retrieve as kb_retrieve,
)
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


def _empty_prescription(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """空处方：无匹配案例/解析失败时返回，交由医生填写"""
    return {
        "disease": diagnosis.get("disease", ""),
        "syndrome": diagnosis.get("syndrome", ""),
        "drugList": [],
        "instruction": {},
    }


async def prescribe_from_kb(
    diagnosis: dict[str, Any],
    results: list[dict[str, Any]],
    patient_info: dict[str, Any] | None = None,
    chief_complaint: str = "",
    inquiry_info: dict[str, Any] | None = None,
    orchestrator: LLMOrchestrator | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """从知识库选案并「原封不动」采用其处方

    检索由调用方完成（graph 节点，性别+年龄±10 过滤），本函数只负责：
    预解析可用的【处方】案例 → LLM 选案（CaseSelectionResult）→
    代码解析处方 + LLM 生成用法 → 组装。
    无匹配 / 解析失败 / 选案异常 → 返回空处方（drugList=[]），由医生填写。

    Args:
        diagnosis: 辨证结果（包含 disease, syndrome）
        results: expert 库检索到的候选案例（[{text,...}]，可空）
        patient_info: 患者信息（性别/年龄/既往史/过敏史）
        chief_complaint: 主诉
        inquiry_info: 追问采集信息
        orchestrator: LLM 编排器实例

    Returns:
        (prescription_dict, reason_dict|None)
        reason 供写 Redis 审计（不展示给用户）；无匹配时 reason.matched=False
    """
    if orchestrator is None:
        orchestrator = LLMOrchestrator()
    patient_info = patient_info or {}

    empty = _empty_prescription(diagnosis)

    if not results:
        return empty, {"matched": False, "analysis": "知识库无匹配案例，处方待医生填写"}

    # 1) 预解析：只保留可解析【处方】的候选
    candidates: list[dict[str, Any]] = []
    for r in results:
        parsed = parse_case_prescription(r.get("text") or "")
        if parsed:
            candidates.append({
                "text": r["text"],
                "drugList": parsed[0],
                "doseNumber": parsed[1],
            })
    if not candidates:
        return empty, {"matched": False, "analysis": "候选案例无可用处方，处方待医生填写"}

    # 3) LLM 选案
    prompt = build_case_selection_prompt(
        patient_info=patient_info,
        diagnosis=diagnosis,
        chief_complaint=chief_complaint,
        inquiry_info=inquiry_info,
        candidates=[c["text"] for c in candidates],
    )
    try:
        sel = await orchestrator.ainvoke_structured(
            CaseSelectionResult,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请从以上候选案例中选择最匹配的一个。"},
            ],
        )
    except Exception as e:
        logger.warning("选案 LLM 异常: %s", e)
        sel = None
    if sel is None or not (1 <= sel.selected_index <= len(candidates)):
        return empty, {"matched": False, "analysis": "选案失败，处方待医生填写"}

    chosen = candidates[sel.selected_index - 1]
    drug_list = chosen["drugList"]
    dose_number = chosen["doseNumber"]

    # 4) LLM 生成用法（剂数强制取案例）
    try:
        inst = await orchestrator.ainvoke_structured(
            InstructionInfo,
            [
                {"role": "system", "content": build_prescription_instruction_prompt(
                    patient_info=patient_info,
                    diagnosis=diagnosis,
                    drug_list=drug_list,
                    dose_number=dose_number,
                    selected_case=chosen["text"],
                )},
                {"role": "user", "content": "请生成用法说明。"},
            ],
        )
    except Exception as e:
        logger.warning("用法生成异常: %s", e)
        inst = None
    inst_dict = inst.model_dump() if inst else InstructionInfo(doseNumber=dose_number).model_dump()
    inst_dict["doseNumber"] = dose_number or inst_dict.get("doseNumber", "")

    prescription = {
        "disease": diagnosis.get("disease", ""),
        "syndrome": diagnosis.get("syndrome", ""),
        "drugList": drug_list,
        "instruction": inst_dict,
    }
    case_header = (chosen["text"].strip().splitlines() or [""])[0]
    reason = {
        "matched": True,
        "selected_index": sel.selected_index,
        "analysis": sel.analysis,
        "case_header": case_header,
        "drug_count": len(drug_list),
        "dose_number": dose_number,
        "disease": prescription["disease"],
        "syndrome": prescription["syndrome"],
    }
    return prescription, reason
