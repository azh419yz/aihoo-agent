"""线下病历多模态分析

使用 qwen3.8-max（统一多模态模型）分析患者上传的线下病历/处方照片，
提取诊断信息、处方信息、患者基本信息等。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)

MEDICAL_RECORD_PROMPT = """请分析这张病历或处方照片，尽可能提取以下信息。
没有观察到的信息留空字符串，不要编造。

请严格按照以下 JSON 格式输出：

```json
{
  "diagnosis": "诊断结果（如'不寐病'）",
  "symptoms": "症状描述（原始文本）",
  "current_symptoms": "当前症状（患者自述的症状表现）",
  "medications": "处方药物",
  "dosage": "用法用量",
  "patient_gender": "患者性别（male/female，不确定则留空）",
  "patient_age": "患者年龄（数字，不确定则留空）",
  "chief_complaint": "主诉（如'失眠'，不确定则留空）",
  "allergy_history": "过敏史（如'青霉素过敏'，不确定则留空）",
  "past_medical_history": "既往病史（如'高血压'，不确定则留空）",
  "summary": "综合摘要（50字以内）"
}
```"""


async def analyze_medical_record(
        image_url: str,
        orchestrator: LLMOrchestrator,
) -> dict[str, Any]:
    """分析线下病历照片

    Args:
        image_url: 病历照片 URL
        orchestrator: LLM 编排器实例

    Returns:
        提取的病历信息
    """
    logger.info("分析线下病历: %s", image_url)

    try:
        messages = [
            {"role": "system", "content": "你是一个专业的医疗文档分析助手，擅长从病历和处方照片中提取信息。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MEDICAL_RECORD_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        response = await orchestrator.chat_with_vl(messages)
        return _parse_json_response(response, {
            "diagnosis": "",
            "symptoms": "",
            "current_symptoms": "",
            "medications": "",
            "dosage": "",
            "patient_gender": "",
            "patient_age": "",
            "chief_complaint": "",
            "allergy_history": "",
            "past_medical_history": "",
            "summary": "",
        })
    except Exception as e:
        logger.error("病历分析失败: %s", e)
        return {
            "diagnosis": "",
            "symptoms": "",
            "current_symptoms": "",
            "medications": "",
            "dosage": "",
            "patient_gender": "",
            "patient_age": "",
            "chief_complaint": "",
            "allergy_history": "",
            "past_medical_history": "",
            "summary": f"病历分析失败: {e}",
        }


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
