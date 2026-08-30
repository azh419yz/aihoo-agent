"""百炼知识库检索

使用百炼知识库检索 API：
  POST https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1/indices/knowledge/search

优化点：
  - 复用 httpx 连接池，避免每次新建客户端
  - 支持查询词丰富（结合 TCM 辨病辨证匹配结果提高命中率）
  - 专家知识库差异化检索策略（优先搜索相似病例）
  - 增强结果格式化（标注来源、结构化展示）
  - 临时故障自动重试（指数退避）
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# Agent ID 映射
AGENT_IDS = {
    "general": settings.KNOWLEDGE_BASE_GENERAL_AGENT_ID,
    "expert": settings.KNOWLEDGE_BASE_EXPERT_AGENT_ID,
}

# 知识库显示名称
_KB_LABELS = {
    "general": "通用中医知识库",
    "expert": "名医经验知识库",
}

# 模块级共享 httpx 客户端（连接池复用）
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """获取共享 httpx 客户端（懒加载，连接复用以减少开销）"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    return _client


# ============================================================
# 查询词构建
# ============================================================


def build_enhanced_query(
        query: str,
        kb_type: str = "general",
        diseases: list[str] | None = None,
        syndromes: list[str] | None = None,
        gender: str | None = None,
        age: int | None = None,
) -> str:
    """构建增强查询词，提高知识库命中率

    根据知识库类型采用不同策略：
      - 通用知识库(理论说明)：以原症状查询为主，附加疾病/证型/性别/年龄标签
      - 专家知识库(病例经验)：优先用疾病证型搜索相似病例，再附加症状/性别/年龄

    Args:
        query: 原始查询（患者症状描述）
        kb_type: 知识库类型 (general / expert)
        diseases: 匹配到的疾病名称列表（如 ["阳痿", "淋证"]）
        syndromes: 匹配到的证型名称列表（如 ["湿热蕴结", "气滞血瘀"]）
        gender: 可选，患者性别（"male"/"female"），帮助匹配相似病例
        age: 可选，患者年龄，帮助匹配相似病例

    Returns:
        增强后的查询词
    """
    if not query.strip():
        return query

    has_context = bool(diseases or syndromes or gender or age is not None)

    if kb_type == "expert":
        # 专家知识库：存储的是完整病例，优先搜索相似病例
        # 用"疾病+证型"作为主搜索，再附加性别/年龄/症状让匹配更精准
        parts: list[str] = []

        if diseases:
            parts.append("疾病：" + "、".join(diseases[:3]))
        if syndromes:
            parts.append("证型：" + "、".join(syndromes[:3]))
        if gender:
            parts.append("性别：" + ("男" if gender == "male" else "女"))
        if age is not None:
            parts.append(f"年龄：约{age}岁")

        if parts and query:
            parts.append(query)
        elif query:
            parts.append(query)

        return "，".join(parts)

    else:
        # 通用知识库：存储的是中医理论知识，原症状查询为主
        if not has_context:
            return query

        tags = (diseases or [])[:3] + (syndromes or [])[:3]
        if gender:
            tags.append("男" if gender == "male" else "女")
        if age is not None:
            tags.append(f"{age}岁")
        tag_str = "、".join(tags)
        return f"{query}（{tag_str}）"


# ============================================================
# 核心检索
# ============================================================


async def retrieve(
        query: str,
        knowledge_base: str = "general",
        top_k: int = 5,
        diseases: list[str] | None = None,
        syndromes: list[str] | None = None,
        gender: str | None = None,
        age: int | None = None,
) -> list[dict[str, Any]]:
    """检索百炼知识库

    Args:
        query: 检索问题（患者症状描述）
        knowledge_base: 知识库类型 ("general" / "expert" / "both")
        top_k: 返回数量上限
        diseases: 可选，匹配到的疾病名称（用于增强查询）
        syndromes: 可选，匹配到的证型名称（用于增强查询）
        gender: 可选，患者性别，附加到查询词提高命中率
        age: 可选，患者年龄，附加到查询词提高命中率

    Returns:
        检索结果列表，每项包含 text / score / source 字段
    """
    if not settings.DASHSCOPE_API_KEY:
        logger.warning("DASHSCOPE_API_KEY 未配置，跳过知识库检索")
        return []

    if not settings.BAILIAN_WORKSPACE_ID:
        logger.warning("BAILIAN_WORKSPACE_ID 未配置，跳过知识库检索")
        return []

    if knowledge_base == "both":
        # 相同查询词分别检索两个库
        enhanced_general = build_enhanced_query(query, "general", diseases, syndromes, gender, age)
        enhanced_expert = build_enhanced_query(query, "expert", diseases, syndromes, gender, age)

        general_results = await _search_knowledge(enhanced_general, "general")
        expert_results = await _search_knowledge(enhanced_expert, "expert")

        merged = _merge_results(general_results, expert_results, top_k)
        logger.debug(
            "知识库检索 [both] query=%s 通用=%d 专家=%d 合并=%d",
            query[:60], len(general_results), len(expert_results), len(merged),
        )
        return merged

    # 单库检索
    enhanced = build_enhanced_query(query, knowledge_base, diseases, syndromes, gender, age)
    results = await _search_knowledge(enhanced, knowledge_base)

    logger.debug(
        "知识库检索 [%s] query=%s 增强后=%s 结果=%d",
        knowledge_base, query[:60], enhanced[:80], len(results),
    )
    return results[:top_k]


def _parse_case_age_gender(text: str) -> tuple[str | None, int | None]:
    """从 expert 库病例文本解析 性别/年龄

    兼容新旧格式：'男｜33岁'、'女/38岁'、'患者: 33岁/男'。
    Returns: (性别中文, 年龄)；无法解析返回 (None, None)
    """
    if not text:
        return None, None
    m = re.search(r'([男女])\s*[｜|/]\s*(\d+)\s*岁', text)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r'(\d+)\s*岁\s*[｜|/]\s*([男女])', text)
    if m:
        return m.group(2), int(m.group(1))
    return None, None


async def retrieve_with_filter(
    query: str,
    gender: str | None = None,
    age: int | None = None,
    age_range: int = 10,
    top_k: int = 3,
    fetch_k: int = 30,
    syndromes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """从 expert 知识库检索相似病例，检索后按 性别精确 + 年龄±age_range 过滤

    RAG 检索是相似度，性别/年龄用检索后过滤硬性保证：
      1. 先取 fetch_k 条候选
      2. 解析每条病例的 性别/年龄，过滤：性别一致 + 年龄在 ±age_range 内
      3. 过滤后取前 top_k（要求过滤时，无法解析性别/年龄的病例剔除）

    Args:
        query: 检索词（疾病/证型/主诉/追问）
        gender: 患者性别（male/female）
        age: 患者年龄
        age_range: 年龄允许上下差（默认 10）
        top_k: 过滤后返回条数
        fetch_k: 先取的候选条数
        syndromes: 可选，证型标签（增强查询）
    """
    results = await retrieve(
        query, knowledge_base="expert", top_k=fetch_k, syndromes=syndromes
    )
    gender_cn = {"male": "男", "female": "女"}.get(gender or "")
    filtered = []
    for r in results:
        g, a = _parse_case_age_gender(r.get("text") or "")
        if gender_cn and g != gender_cn:
            continue
        if age is not None:
            if a is None:
                continue  # 要求年龄过滤时，无法解析的病例剔除
            if abs(a - age) > age_range:
                continue
        filtered.append(r)
    return filtered[:top_k]


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
async def _search_knowledge(query: str, kb_type: str) -> list[dict[str, Any]]:
    """调用知识库检索 API（自动重试临时故障）"""
    agent_id = AGENT_IDS.get(kb_type)
    if not agent_id:
        logger.warning("知识库 %s 未配置 Agent ID，跳过检索", kb_type)
        return []

    url = (
        f"https://{settings.BAILIAN_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com"
        "/api/v1/indices/knowledge/search"
    )

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "agent_id": agent_id,
    }

    client = _get_client()
    response = await client.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        logger.error(
            "知识库检索失败 [%s]: status=%d body=%s",
            kb_type, response.status_code, response.text[:300],
        )
        if response.status_code >= 500:
            # 5xx 让 tenacity 重试
            response.raise_for_status()
        return []

    data = response.json()
    return _parse_results(data)


# ============================================================
# 结果解析
# ============================================================


def _parse_results(data: Any) -> list[dict[str, Any]]:
    """统一解析检索结果

    兼容不同响应格式：
      - 顶层是 list → 直接解析每个 item
      - 顶层是 dict → 尝试 data/content/results/chunks 等常见 key
    """
    raw: list[dict[str, Any]] = []

    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        # 响应格式: { "data": { "nodes": [...] } }
        inner = data.get("data") or data
        if isinstance(inner, dict):
            nodes = inner.get("nodes") or inner.get("results") or inner.get("chunks") or []
            if isinstance(nodes, list):
                raw = nodes
            else:
                # 兜底：遍历已知 key
                for key in ("data", "content", "results", "chunks", "documents", "records"):
                    if key in inner and isinstance(inner[key], list):
                        raw = inner[key]
                        break
        if not raw:
            raw = [data]

    results: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # 优先取 metadata.content（百炼新 API 格式）
        meta = item.get("metadata") or {}
        text = (
                meta.get("content")
                or meta.get("text")
                or item.get("text")
                or item.get("content")
                or item.get("chunk")
                or item.get("snippet")
                or ""
        )
        if text and isinstance(text, str) and text.strip():
            results.append({
                "text": text.strip(),
                "score": item.get("score", 0) or meta.get("score", 0),
            })
    return results


# ============================================================
# 结果合并
# ============================================================


def _merge_results(
        general: list[dict[str, Any]],
        expert: list[dict[str, Any]],
        top_k: int,
) -> list[dict[str, Any]]:
    """合并两个知识库的结果

    - 按来源标记 source 字段
    - 按 score 降序排列
    - 去重后截断
    """
    seen = set()
    merged: list[dict[str, Any]] = []

    # 标记来源
    for r in general:
        r["source"] = "general"
    for r in expert:
        r["source"] = "expert"

    # 合并并按 score 排序（没有 score 的排在后面）
    combined = general + expert
    combined.sort(key=lambda x: -x.get("score", 0))

    for r in combined:
        text = r.get("text", "")
        if text and text not in seen:
            seen.add(text)
            merged.append(r)

    return merged[:top_k]


# ============================================================
# 格式化
# ============================================================


def format_retrieval_context(results: list[dict[str, Any]]) -> str:
    """将检索结果格式化为 LLM 可用的结构化上下文文本

    增强展示：
      - 标注知识库来源（通用中医 / 名医经验）
      - 结构化分段
    """
    if not results:
        return ""

    chunks: list[str] = []
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        if not text:
            continue

        # 来源标签
        source = result.get("source", "")
        source_label = _KB_LABELS.get(source) or "知识库"

        chunks.append(f"[{source_label}]\n{text}")

    return "\n\n".join(chunks)
