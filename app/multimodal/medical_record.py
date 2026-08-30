"""线下病历多模态分析

使用 qwen3.8-max（统一多模态模型）分析患者上传的线下病历照片。
支持多张图片；只提取患者基础信息（姓名/性别/年龄/身高/职业/体重），
不解析诊断、症状、处方等医疗内容（避免医疗内容影响男科辨证）。

"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)

MEDICAL_RECORD_PROMPT = """请分析这些病历照片，只提取患者的基础信息。
不要提取诊断、症状、处方等医疗内容，也不要对病情做任何分析。
没有观察到的信息留空字符串，不要编造。

请严格按照以下 JSON 格式输出：

```json
{
  "patient_name": "患者姓名（不确定则留空）",
  "patient_gender": "患者性别（male/female，不确定则留空）",
  "patient_age": "患者年龄（数字，不确定则留空）",
  "patient_height": "身高（如'175cm'，不确定则留空）",
  "patient_occupation": "职业（如'程序员'，不确定则留空）",
  "patient_weight": "体重（如'70kg'，不确定则留空）"
}
```"""


async def analyze_medical_record(
        image_urls: list[str],
        orchestrator: LLMOrchestrator,
) -> dict[str, Any]:
    """分析线下病历照片（支持多张图片）

    多张图片一起送入模型，便于跨页核对基础信息（如第一页姓名、第二页年龄）。

    Args:
        image_urls: 病历照片 URL 列表（支持多张）
        orchestrator: LLM 编排器实例

    Returns:
        提取的病历基础信息
    """
    logger.info("分析线下病历: %s", image_urls)

    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": MEDICAL_RECORD_PROMPT}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的医疗文档信息提取助手，只从病历照片中提取患者基础信息。",
            },
            {"role": "user", "content": content},
        ]
        response = await orchestrator.chat_with_vl(messages)
        return _parse_json_response(response, {
            "patient_name": "",
            "patient_gender": "",
            "patient_age": "",
            "patient_height": "",
            "patient_occupation": "",
            "patient_weight": "",
        })
    except Exception as e:
        logger.error("病历分析失败: %s", e)
        return {
            "patient_name": "",
            "patient_gender": "",
            "patient_age": "",
            "patient_height": "",
            "patient_occupation": "",
            "patient_weight": "",
            "error": f"病历分析失败: {e}",
        }


def format_medical_record_basic_info(record_data: dict) -> str:
    """把病历提取的基础信息格式化为展示文本（供确认展示）"""
    lines = []
    if record_data.get("patient_name"):
        lines.append(f"- 姓名: {record_data['patient_name']}")
    gender = record_data.get("patient_gender")
    if gender:
        display = "男" if gender == "male" else ("女" if gender == "female" else gender)
        lines.append(f"- 性别: {display}")
    if record_data.get("patient_age"):
        lines.append(f"- 年龄: {record_data['patient_age']}岁")
    if record_data.get("patient_height"):
        lines.append(f"- 身高: {record_data['patient_height']}")
    if record_data.get("patient_occupation"):
        lines.append(f"- 职业: {record_data['patient_occupation']}")
    if record_data.get("patient_weight"):
        lines.append(f"- 体重: {record_data['patient_weight']}")
    return "\n".join(lines) if lines else "- （未识别到基础信息）"


def extract_oss_urls(message: str) -> list[str]:
    """从消息中提取 OSS 图片 URL

    支持格式: "oss地址: https://xxx.com/img.jpg"
    """
    urls = re.findall(r'https?://[^\s，。、\'"]+(?:jpg|jpeg|png|gif|webp)', message, re.IGNORECASE)
    return urls


def _parse_json_response(response: str, default: dict) -> dict:
    """从 LLM 响应中解析 JSON"""
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

    logger.warning("无法从病历分析响应解析 JSON")
    return default
