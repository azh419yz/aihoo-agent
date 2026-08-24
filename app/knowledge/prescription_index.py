"""处方索引 — 从 MySQL tcm_prescription 表加载历史处方数据

功能：
  1. 启动时从 MySQL 加载历史处方
  2. 按疾病+证型模糊匹配相似处方
  3. 格式化为 LLM 处方生成的结构化参考上下文

匹配策略（逐层降级）：
  - 精确匹配 disease + syndrome（完全一致）
  - 疾病匹配 disease only（同病不同证）
  - 证型匹配 syndrome only（同证不同病）

用法：
  await prescription_index.load(engine)     # engine = AsyncEngine
  ctx = prescription_index.format_context("阳痿", "肾阳虚证")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


class PrescriptionIndex:
    """处方索引 — 从 MySQL 加载历史处方数据，支持按病/证检索"""

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        # 预构建索引：disease → list of prescriptions
        self._by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # syndrome → list of prescriptions
        self._by_syndrome: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # (disease, syndrome) → list of prescriptions（精确匹配）
        self._by_both: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def count(self) -> int:
        return len(self._data)

    async def load(self, engine: AsyncEngine) -> None:
        """从 MySQL tcm_prescription 表加载处方数据并构建索引

        Args:
            engine: SQLAlchemy 异步引擎（来自 mysql_client.engine）
        """
        if not engine:
            logger.warning("PrescriptionIndex: engine 未就绪，跳过加载")
            return

        try:
            async with engine.connect() as conn:
                sql = text(
                    "SELECT disease, syndrome, herbs, dosage, age, gender "
                    "FROM tcm_prescription "
                    "ORDER BY disease, syndrome"
                )
                result = await conn.execute(sql)
                rows = result.fetchall()
        except Exception as e:
            logger.warning("PrescriptionIndex: 加载失败: %s", e)
            return

        # 解析行数据并构建索引
        for row in rows:
            mapping = row._mapping
            disease = (mapping.get("disease") or "").strip()
            syndrome = (mapping.get("syndrome") or "").strip()
            herbs_raw = mapping.get("herbs")
            dosage = (mapping.get("dosage") or "") or ""
            age = mapping.get("age")
            gender = (mapping.get("gender") or "") or ""

            # 解析 herbs JSON
            herbs: list[str] = []
            if herbs_raw:
                if isinstance(herbs_raw, str):
                    try:
                        herbs = json.loads(herbs_raw)
                    except (json.JSONDecodeError, TypeError):
                        herbs = []
                elif isinstance(herbs_raw, (list, tuple)):
                    herbs = list(herbs_raw)

            entry = {
                "disease": disease,
                "syndrome": syndrome,
                "herbs": herbs,
                "dosage": dosage,
                "age": age,
                "gender": gender,
            }

            self._data.append(entry)
            if disease:
                self._by_disease[disease].append(entry)
            if syndrome:
                self._by_syndrome[syndrome].append(entry)
            if disease and syndrome:
                self._by_both[(disease, syndrome)].append(entry)

        self._loaded = True
        logger.info(
            "PrescriptionIndex: 从 MySQL 加载 %d 条处方数据（%d 疾病, %d 证型）",
            len(self._data),
            len(self._by_disease),
            len(self._by_syndrome),
        )

    def lookup(
            self,
            disease: str,
            syndrome: str,
            top_k: int = 3,
            gender: str | None = None,
            age: int | None = None,
    ) -> list[dict[str, Any]]:
        """按疾病+证型查找相似处方案例，支持性别年龄过滤

        优先级（保留原有三层结构）：
          1. 精确匹配 disease + syndrome
          2. 仅同 disease（不同 syndrome）
          3. 仅同 syndrome（不同 disease）

        在每层结果中按以下规则过滤（逐层放宽）：
          - 性别精确匹配（男/女）
          - 年龄 ±5 岁优先 → ±10 岁 → 不限年龄
          - 如果该层严格过滤后不足 top_k，放宽条件

        Args:
            disease: 辨病名称
            syndrome: 证型名称
            top_k: 返回最大数量
            gender: 可选，患者性别（"male"/"female"），精确匹配
            age: 可选，患者年龄，±5 岁优先

        Returns:
            匹配的处方列表，每条包含 herbs/dosage/age/gender 等字段
        """
        if not self._loaded:
            return []

        candidates = self._get_all_candidates(disease, syndrome)
        if not candidates:
            return []

        return self._filter_with_relaxation(
            candidates, gender=gender, age=age, top_k=top_k,
        )

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _get_all_candidates(
            self, disease: str, syndrome: str,
    ) -> list[dict[str, Any]]:
        """从三层索引收集所有候选处方（已去重）"""
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        def _add(entries: list[dict]) -> None:
            for e in entries:
                key = json.dumps(e.get("herbs", []), ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    results.append(e)

        # 1. 精确匹配
        exact = self._by_both.get((disease, syndrome), [])
        _add(exact)

        # 2. 同 disease（但排除已在精确匹配中的）
        same_disease = [
            e for e in self._by_disease.get(disease, [])
            if (e.get("disease"), e.get("syndrome")) != (disease, syndrome)
        ]
        _add(same_disease)

        # 3. 同 syndrome（但排除已出现过的）
        same_syndrome = self._by_syndrome.get(syndrome, [])
        _add(same_syndrome)

        return results

    @staticmethod
    def _filter_candidates(
            candidates: list[dict[str, Any]],
            gender: str | None = None,
            age: int | None = None,
            age_range: int | None = None,
    ) -> list[dict[str, Any]]:
        """过滤候选列表：性别精确匹配 + 年龄范围过滤"""
        if gender is None and age is None:
            return list(candidates)

        result = []
        for c in candidates:
            # 性别过滤（精确匹配）
            if gender and c.get("gender") and c["gender"] != gender:
                continue
            # 年龄过滤（±age_range）
            if age is not None and age_range is not None:
                c_age = c.get("age")
                if c_age is not None and abs(c_age - age) > age_range:
                    continue
            result.append(c)
        return result

    def _filter_with_relaxation(
            self,
            candidates: list[dict[str, Any]],
            gender: str | None = None,
            age: int | None = None,
            top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """逐层放宽条件的过滤

        尝试策略：
          1. 性别精确 + 年龄 ±5
          2. 性别精确 + 年龄 ±10
          3. 性别精确 + 不限年龄
          4. 不限性别 + 不限年龄（原始逻辑）

        每次用剩余结果填充直到满 top_k。
        """
        if gender is None and age is None:
            return candidates[:top_k]

        result: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def _add_dedup(entries: list[dict]) -> None:
            for e in entries:
                key = json.dumps(e.get("herbs", []), ensure_ascii=False)
                if key not in seen_keys and len(result) < top_k:
                    seen_keys.add(key)
                    result.append(e)

        # 策略 1：性别精确 + 年龄 ±5
        if gender or age is not None:
            filtered = self._filter_candidates(candidates, gender=gender, age=age, age_range=5)
            _add_dedup(filtered)

        # 策略 2：性别精确 + 年龄 ±10
        if len(result) < top_k and age is not None:
            filtered = self._filter_candidates(candidates, gender=gender, age=age, age_range=10)
            _add_dedup(filtered)

        # 策略 3：性别精确 + 不限年龄
        if len(result) < top_k and gender:
            filtered = self._filter_candidates(candidates, gender=gender, age=None)
            _add_dedup(filtered)

        # 策略 4：原始逻辑（不限性别/年龄）
        if len(result) < top_k:
            _add_dedup(candidates)

        return result[:top_k]

    def get_similar_dosages(
            self,
            disease: str,
            syndrome: str,
            gender: str | None = None,
            age: int | None = None,
    ) -> list[str]:
        """获取同一病证的常见剂量范围（供 LLM 参考）"""
        entries = self.lookup(disease, syndrome, top_k=5, gender=gender, age=age)
        dosages = []
        for e in entries:
            d = e.get("dosage", "")
            if d and d != "\\N" and d not in dosages:
                dosages.append(d)
        return dosages

    def format_context(
            self,
            disease: str,
            syndrome: str,
            top_k: int = 3,
            gender: str | None = None,
            age: int | None = None,
    ) -> str:
        """格式化为 LLM 处方生成的结构化参考上下文

        Args:
            disease: 辨病名称
            syndrome: 证型名称
            top_k: 展示的最大案例数
            gender: 可选，患者性别，用于匹配相似案例
            age: 可选，患者年龄，用于匹配相似案例

        Returns:
            格式化的 Markdown 字符串，无匹配时返回空字符串
        """
        matches = self.lookup(disease, syndrome, top_k=top_k, gender=gender, age=age)
        if not matches:
            return ""

        parts = [
            "## 历史处方案例参考",
            "> 以下是与该病证相似的历史处方，供处方生成参考。",
        ]

        for i, m in enumerate(matches, 1):
            herbs = m.get("herbs", [])
            dosage = m.get("dosage", "")
            age = m.get("age", "")
            gender = m.get("gender", "")

            # 过滤无效值（原始 Excel 的 \N 标记）
            if dosage == "\\N":
                dosage = ""
            if age == "\\N":
                age = ""

            herb_str = "、".join(herbs[:15])  # 最多展示 15 味药
            if len(herbs) > 15:
                herb_str += "…"

            info_parts = []
            if age:
                info_parts.append(f"{age}岁")
            if gender:
                info_parts.append("男" if gender == "male" else "女" if gender == "female" else gender)
            patient_info = "（" + "、".join(info_parts) + "）" if info_parts else ""

            parts.append(f"\n**案例 {i}**{patient_info}：")
            parts.append(f"  处方：{herb_str}")
            if dosage:
                parts.append(f"  剂数：{dosage} 剂")

        parts.append(
            "\n> **注意**: 以上为相似案例参考，最终处方需结合患者个体情况调整。"
        )
        return "\n".join(parts)


# 全局单例
prescription_index = PrescriptionIndex()
