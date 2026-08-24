"""舌照/面照多模态分析

使用 qwen3.8-max（统一多模态模型，经 orchestrator.chat_with_vl）
分析患者舌照和面照，提取舌象/面色特征。

支持 URL 和 base64 编码的图片。

舌诊分析使用增强版提示词，包含证型提示和养生建议。
提供结构化输出（TongueAnalysisResult）+ 关键词回退解析。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.orchestrator import LLMOrchestrator
from app.agent.structured_output import TongueAnalysisResult

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# 增强版舌诊提示词（来自专家经验）
# ----------------------------------------------------------------

TONGUE_PROMPT = """你是一位经验丰富的中医舌诊专家。请根据用户提供的舌象图片进行专业分析。

分析要点：
1. 舌色：观察舌质颜色（淡红、红、绛红、紫暗、淡白等）
2. 舌形：观察舌体形态（胖大、瘦薄、齿痕、裂纹、芒刺等）
3. 苔色：观察舌苔颜色（白、黄、灰、黑等）
4. 苔质：观察舌苔质地（薄、厚、腻、燥、滑、剥落等）
5. 舌态：观察舌体动态（强硬、痿软、颤动、歪斜等）

请根据以上观察，严格按照以下 JSON 格式输出：

```json
{
  "tongue_color": "舌色（淡红/红/绛/紫/淡白等）",
  "tongue_shape": "舌形（胖大/瘦薄/齿痕/裂纹等）",
  "coating_color": "苔色（白/黄/灰/黑等）",
  "coating_texture": "苔质（薄/厚/腻/燥/滑/剥落等）",
  "analysis": "综合舌诊分析（100字以内，描述舌象特征和中医解释）",
  "syndrome_hints": ["可能的证型提示1", "可能的证型提示2"]
}
```

注意：
- 分析应客观准确，基于图像实际情况
- 证型提示仅供参考，不作为诊断依据
- 如图片质量不佳，请说明
- syndrome_hints 数组至少包含1个证型提示，最多3个"""

FACE_PROMPT = """请分析这张面照，严格按照以下格式输出JSON（不要包含markdown代码块标记）：

```json
{
  "face_color": "面色（红润/萎黄/苍白/晦暗/潮红/青紫等）",
  "complexion": "光泽（有光泽/少光泽/无光泽）",
  "lip_color": "唇色（淡红/红/紫暗/淡白/干裂等）",
  "description": "综合描述（100字以内）"
}
```

注意：如果没有观察到某个特征，留空字符串即可，不要编造。"""


# ============================================================
# 舌诊分析
# ============================================================


async def analyze_tongue(
        image_url: str,
        orchestrator: LLMOrchestrator,
        additional_info: str | None = None,
) -> TongueAnalysisResult:
    """分析舌照，返回结构化结果

    Args:
        image_url: 图片 URL 或 base64 编码
        orchestrator: LLM 编排器实例
        additional_info: 可选的患者补充信息（症状描述等）

    Returns:
        TongueAnalysisResult: 结构化的舌诊分析结果
    """
    logger.info("分析舌照: %s...", image_url[:80])

    try:
        # 构建带图片的提示
        prompt = TONGUE_PROMPT
        if additional_info:
            prompt += f"\n\n患者补充信息：{additional_info}"

        messages = _build_multimodal_message(image_url, prompt)
        response = await orchestrator.chat_with_vl(messages)

        # 尝试结构化解析
        result = _parse_structured_tongue_response(response)
        if result is not None:
            return result

        # JSON 解析失败 → 降级为关键词解析
        logger.info("舌照 JSON 解析失败，使用关键词回退")
        return _parse_response_to_result(response)

    except Exception as e:
        logger.error("舌照分析失败: %s", e)
        return TongueAnalysisResult(
            tongue_color="",
            tongue_shape="",
            coating_color="",
            coating_texture="",
            analysis=f"舌照分析失败：{e}。请确保图片清晰，光线充足。",
            syndrome_hints=[],
        )


# ============================================================
# 面部分析
# ============================================================


async def analyze_face(
        image_url: str,
        orchestrator: LLMOrchestrator,
) -> dict[str, Any]:
    """分析面照

    Args:
        image_url: 图片 URL 或 base64 编码
        orchestrator: LLM 编排器实例

    Returns:
        {"face_color", "complexion", "lip_color", "description"}
    """
    logger.info("分析面照: %s...", image_url[:80])

    try:
        messages = _build_multimodal_message(image_url, FACE_PROMPT)
        response = await orchestrator.chat_with_vl(messages)
        return _parse_face_json(response)
    except Exception as e:
        logger.error("面照分析失败: %s", e)
        return {
            "face_color": "",
            "complexion": "",
            "lip_color": "",
            "description": f"面照分析失败: {e}",
        }


# ============================================================
# 批量分析入口
# ============================================================


async def analyze_tongue_images(
        image_urls: list[str],
        orchestrator: LLMOrchestrator,
        patient_context: str | None = None,
) -> list[dict[str, Any]]:
    """批量分析舌照（支持并发）

    Args:
        image_urls: 舌照 URL 列表
        orchestrator: LLM 编排器实例
        patient_context: 可选的患者症状描述

    Returns:
        舌照分析结果列表
    """
    import asyncio

    futures = [
        analyze_tongue(url, orchestrator, additional_info=patient_context)
        for url in image_urls
    ]
    results = await asyncio.gather(*futures) if futures else []
    return [r.model_dump() if r else {} for r in results]


async def analyze_face_images(
        image_urls: list[str],
        orchestrator: LLMOrchestrator,
) -> list[dict[str, Any]]:
    """批量分析面照（支持并发）

    Args:
        image_urls: 面照 URL 列表
        orchestrator: LLM 编排器实例

    Returns:
        面照分析结果列表
    """
    import asyncio

    futures = [
        analyze_face(url, orchestrator)
        for url in image_urls
    ]
    results = await asyncio.gather(*futures) if futures else []
    return list(results)


# ============================================================
# 辅助方法
# ============================================================


def _build_multimodal_message(image_url: str, prompt: str) -> list:
    """构建多模态消息（OpenAI-compatible 格式，qwen3.8-max 统一走）

    兼容 URL 和 base64 两种图片格式。
    """
    # 判断图片格式
    if image_url.startswith("data:"):
        # base64 格式: data:image/jpeg;base64,xxxx
        url = image_url
    elif image_url.startswith("http"):
        # URL 格式
        url = image_url
    else:
        # 假设是纯 base64 字符串
        url = f"data:image/jpeg;base64,{image_url}"

    image_content = {"type": "image_url", "image_url": {"url": url}}

    return [
        {"role": "system", "content": "你是一个专业的中医诊断助手，擅长舌诊和面诊分析。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content,
            ],
        },
    ]


def _parse_structured_tongue_response(response: str) -> TongueAnalysisResult | None:
    """从 LLM 响应中尝试解析为结构化 TongueAnalysisResult"""
    try:
        # 尝试直接解析 JSON
        data = json.loads(response)
        return TongueAnalysisResult(**data)
    except (json.JSONDecodeError, ValueError):
        pass

    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return TongueAnalysisResult(**data)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_response_to_result(content: str) -> TongueAnalysisResult:
    """关键词回退解析：从 LLM 文本响应中提取特征

    当 JSON 结构化解析失败时，用关键词匹配兜底。
    """
    result = TongueAnalysisResult(
        tongue_color="",
        tongue_shape="",
        coating_color="",
        coating_texture="",
        analysis=content,
        syndrome_hints=[],
    )

    # 尝试提取舌色
    color_keywords = ["淡红", "红", "绛红", "紫暗", "淡白", "青紫"]
    for keyword in color_keywords:
        if keyword in content:
            result.tongue_color = keyword
            break

    # 尝试提取舌形
    shape_keywords = ["胖大", "瘦薄", "齿痕", "裂纹", "芒刺"]
    for keyword in shape_keywords:
        if keyword in content:
            result.tongue_shape = keyword
            break

    # 尝试提取苔色
    coating_color_keywords = ["白苔", "黄苔", "灰苔", "黑苔", "白", "黄", "灰", "黑"]
    for keyword in coating_color_keywords:
        if keyword in content:
            result.coating_color = keyword.replace("苔", "")
            break

    # 尝试提取苔质
    texture_keywords = ["薄苔", "厚苔", "腻苔", "燥苔", "滑苔", "薄", "厚", "腻", "燥", "滑"]
    for keyword in texture_keywords:
        if keyword in content:
            result.coating_texture = keyword.replace("苔", "")
            break

    # 提取证型提示
    syndrome_keywords = [
        "气虚", "血虚", "阴虚", "阳虚", "痰湿", "湿热", "血瘀",
        "气滞", "寒湿", "风热", "风寒", "肝郁", "脾虚", "肾虚",
        "心脾两虚", "肝肾阴虚", "脾肾阳虚", "肝气郁结", "气血两虚",
    ]
    for keyword in syndrome_keywords:
        if keyword in content:
            result.syndrome_hints.append(keyword)

    return result


def _parse_face_json(response: str) -> dict[str, Any]:
    """从 LLM 响应中解析面部分析 JSON"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    logger.warning("无法从响应解析面部分析 JSON")
    return {
        "face_color": "",
        "complexion": "",
        "lip_color": "",
        "description": response[:200],
    }
