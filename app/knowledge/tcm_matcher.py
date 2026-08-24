"""中医辨病辨证匹配器

从 MySQL 加载 tcm_disease / tcm_syndrome 数据到内存，
根据患者症状做关键词匹配，输出可能的疾病和证型。

在 INQUIRY / DIAGNOSIS 阶段为 LLM 提供结构化辨病辨证参考。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings

logger = logging.getLogger(__name__)

_MIN_KEYWORD_LENGTH = 2

# 中医症状同义词映射（口语 → 数据库标准术语）
# 患者描述症状时常用口语化表达，映射到数据库中的标准中医术语以提高匹配率
_SYMPTOM_SYNONYMS: dict[str, set[str]] = {
    # ---- 发热/寒热 ----
    "发烧": {"发热"},
    "发高烧": {"发热", "高热"},
    "低烧": {"低热"},
    "反复发烧": {"发热"},
    "怕冷": {"恶寒", "畏寒"},
    "怕风": {"恶风"},
    "发冷": {"恶寒", "畏寒"},
    "身上冷": {"恶寒", "畏寒"},
    "手脚凉": {"四肢不温"},
    "手脚冷": {"四肢厥冷"},
    "手脚冰凉": {"四肢不温", "四肢厥冷"},
    "忽冷忽热": {"寒热往来"},
    "一阵冷一阵热": {"寒热往来"},
    "怕热": {"恶热"},
    "爱上火": {"内热"},
    "火气大": {"内热"},

    # ---- 出汗 ----
    "出虚汗": {"自汗", "盗汗"},
    "睡着出汗": {"盗汗"},
    "睡觉出汗": {"盗汗"},
    "夜里出汗": {"盗汗"},
    "睡醒出汗": {"盗汗"},
    "汗多": {"自汗", "多汗"},
    "动不动就出汗": {"自汗"},
    "冷汗": {"自汗"},

    # ---- 头部/面部 ----
    "头疼": {"头痛"},
    "脑袋疼": {"头痛"},
    "偏头疼": {"头痛"},
    "前额疼": {"头痛"},
    "后脑勺疼": {"头痛"},
    "头顶疼": {"巅顶痛"},
    "头昏": {"眩晕"},
    "头晕": {"眩晕"},
    "天旋地转": {"眩晕"},
    "晕": {"眩晕"},
    "头发懵": {"头晕", "头重"},
    "头重": {"头重"},
    "脸红": {"面红"},
    "脸黄": {"面色萎黄"},
    "脸白": {"面色苍白"},
    "脸色不好": {"面色萎黄", "面色少华"},
    "眼睛干": {"目涩"},
    "眼干": {"目涩"},
    "眼睛花": {"目眩", "视物模糊"},
    "视力模糊": {"视物模糊"},
    "眼睛红": {"目赤"},
    "眼睛肿": {"目胞浮肿"},
    "耳鸣": {"耳鸣"},
    "耳朵响": {"耳鸣"},
    "耳朵嗡": {"耳鸣"},
    "听力下降": {"耳鸣"},

    # ---- 鼻部 ----
    "流清鼻涕": {"鼻塞流清涕"},
    "流鼻涕": {"鼻塞流涕", "流涕"},
    "流黄鼻涕": {"鼻塞流浊涕"},
    "鼻子堵": {"鼻塞"},
    "鼻子不通": {"鼻塞"},
    "打喷嚏": {"喷嚏"},
    "鼻塞": {"鼻塞"},
    "鼻子干": {"鼻干"},

    # ---- 咽喉/口部 ----
    "嗓子疼": {"咽喉肿痛", "咽痛"},
    "喉咙痛": {"咽喉肿痛", "咽痛"},
    "嗓子干": {"咽干", "口燥咽干"},
    "喉咙干": {"咽干", "口燥咽干"},
    "嗓子痒": {"咽痒"},
    "喉咙痒": {"咽痒"},
    "嗓子有痰": {"咳痰", "痰"},
    "喉咙有异物": {"咽中异物感", "梅核气"},
    "口干": {"口干", "口燥"},
    "口渴": {"口渴", "口干"},
    "总想喝水": {"口渴"},
    "不想喝水": {"口渴不欲饮"},
    "口苦": {"口苦"},
    "嘴里发苦": {"口苦"},
    "口臭": {"口臭"},
    "嘴里有味": {"口臭"},
    "嘴里发甜": {"口甜"},
    "嘴里发黏": {"口黏", "口黏腻"},
    "口干不想喝水": {"口渴不欲饮"},
    "口腔溃疡": {"口疮"},
    "长口疮": {"口疮"},

    # ---- 咳嗽/呼吸 ----
    "咳": {"咳嗽"},
    "干咳": {"干咳"},
    "咳嗽": {"咳嗽"},
    "咳痰": {"咳痰"},
    "有痰": {"咳痰", "痰"},
    "痰多": {"咳痰", "痰多"},
    "痰黄": {"痰黄"},
    "痰白": {"痰白清稀"},
    "痰黏": {"痰黏稠"},
    "痰稠": {"痰黏稠"},
    "咳不出": {"痰黏稠难咳"},
    "喘": {"气喘"},
    "喘不上气": {"气喘", "呼吸困难"},
    "气短": {"气短"},
    "上不来气": {"气喘", "呼吸困难"},
    "胸闷": {"胸闷"},
    "胸口闷": {"胸闷"},

    # ---- 心胸部 ----
    "心慌": {"心悸"},
    "心跳快": {"心悸"},
    "心跳": {"心悸"},
    "心突突": {"心悸"},
    "心绞痛": {"胸痛", "心痛"},
    "胸口疼": {"胸痛"},
    "胸痛": {"胸痛"},
    "胸口闷": {"胸闷"},
    "心口堵": {"胸闷", "胸痹"},

    # ---- 胃/腹部 ----
    "胃疼": {"胃痛", "胃脘痛"},
    "胃痛": {"胃痛"},
    "胃胀": {"胃胀", "脘痞"},
    "胃不舒服": {"胃痛", "胃胀"},
    "不想吃饭": {"纳呆", "纳少"},
    "没胃口": {"纳呆", "纳少"},
    "吃不下": {"纳呆", "纳少"},
    "食欲差": {"纳呆", "纳少"},
    "肚子疼": {"腹痛"},
    "肚子胀": {"腹胀"},
    "腹胀": {"腹胀"},
    "小腹疼": {"少腹痛"},
    "下腹痛": {"少腹痛"},
    "反酸": {"反酸", "吞酸"},
    "烧心": {"反酸", "吞酸"},
    "恶心": {"恶心", "呕恶"},
    "想吐": {"恶心", "呕吐"},
    "呕吐": {"呕吐"},
    "打嗝": {"嗳气"},
    "嗳气": {"嗳气"},

    # ---- 大便 ----
    "拉肚子": {"腹泻"},
    "拉稀": {"腹泻"},
    "拉": {"腹泻"},
    "腹泻": {"腹泻"},
    "水样便": {"腹泻"},
    "大便稀": {"大便溏薄", "大便不成形"},
    "大便不成形": {"大便不成形"},
    "大便黏": {"大便黏滞", "大便粘稠"},
    "大便粘": {"大便黏滞", "大便粘稠"},
    "大便干": {"便秘", "大便干结", "大便干燥"},
    "便秘": {"便秘"},
    "大便困难": {"便秘", "排便困难"},
    "拉不出": {"便秘"},
    "好几天不上厕所": {"便秘"},
    "大便带血": {"便血"},
    "便血": {"便血"},
    "大便黑": {"大便发黑"},
    "大便次数多": {"大便频数"},
    "拉不干净": {"里急后重"},
    "大便不尽": {"里急后重"},
    "大便粗": {"大便干结"},

    # ---- 小便 ----
    "尿多": {"小便频数", "夜尿多"},
    "尿频": {"小便频数"},
    "总上厕所": {"小便频数"},
    "夜尿多": {"夜尿多"},
    "晚上总起夜": {"夜尿多"},
    "尿急": {"小便急迫", "尿急"},
    "尿痛": {"小便涩痛", "淋沥涩痛"},
    "尿黄": {"小便黄"},
    "尿热": {"小便短赤"},
    "尿少": {"小便短少"},
    "尿不出来": {"小便不利"},
    "小便不利": {"小便不利"},
    "尿血": {"尿血"},
    "尿里有泡沫": {"尿浊"},

    # ---- 睡眠 ----
    "睡不着": {"不寐", "失眠"},
    "失眠": {"不寐"},
    "睡不好": {"不寐", "失眠"},
    "入睡难": {"入睡困难"},
    "睡不踏实": {"寐而不酣", "易醒"},
    "容易醒": {"易醒"},
    "醒了睡不着": {"醒后不能再寐"},
    "做梦多": {"多梦"},
    "总做梦": {"多梦"},
    "打呼噜": {"打鼾"},
    "打鼾": {"打鼾"},
    "犯困": {"嗜睡", "困倦"},
    "白天困": {"嗜睡"},
    "睡不醒": {"嗜睡"},

    # ---- 精神/情志 ----
    "没劲": {"神疲", "乏力"},
    "没力气": {"神疲", "乏力"},
    "累": {"神疲", "乏力"},
    "疲劳": {"神疲", "乏力"},
    "浑身没劲": {"神疲", "乏力"},
    "没精神": {"神疲", "乏力", "精神萎靡"},
    "精神不好": {"神疲", "精神萎靡"},
    "不想动": {"乏力", "肢体倦怠"},
    "懒": {"乏力", "少气懒言"},
    "懒得说话": {"少气懒言"},
    "脾气急": {"急躁易怒"},
    "爱发火": {"急躁易怒"},
    "烦躁": {"烦躁", "心烦"},
    "心烦": {"心烦"},
    "心情不好": {"情志抑郁", "情绪不宁"},
    "郁闷": {"情志抑郁"},
    "压抑": {"情志抑郁"},
    "叹气": {"善太息"},
    "总叹气": {"善太息"},
    "焦虑": {"情志抑郁", "心烦"},
    "紧张": {"心悸", "惊惕不安"},

    # ---- 疼痛（通用） ----
    "腰疼": {"腰痛"},
    "腰酸": {"腰膝酸软"},
    "腰膝酸软": {"腰膝酸软"},
    "腰没劲": {"腰膝酸软"},
    "后背疼": {"背痛"},
    "背疼": {"背痛"},
    "肩膀疼": {"肩痛"},
    "脖子疼": {"颈项强痛", "颈痛"},
    "脖子硬": {"颈项强痛"},
    "颈椎不舒服": {"颈项强痛"},
    "关节疼": {"关节疼痛"},
    "关节痛": {"关节疼痛"},
    "膝盖疼": {"膝痛"},
    "腿疼": {"下肢疼痛"},
    "腿肿": {"下肢浮肿"},
    "脚肿": {"足跗浮肿"},
    "全身疼": {"全身酸痛", "身痛"},
    "身上疼": {"身痛", "全身酸痛"},
    "肌肉疼": {"肌肉酸痛"},
    "抽筋": {"抽搐"},
    "腿抽筋": {"转筋"},

    # ---- 妇科 ----
    "月经不调": {"月经不调"},
    "痛经": {"痛经"},
    "月经疼": {"痛经"},
    "月经少": {"月经量少"},
    "月经多": {"月经量多"},
    "月经有块": {"经色紫暗有块"},
    "白带多": {"带下量多"},
    "白带异常": {"带下病"},
    "带下有味": {"带下臭秽"},
    "白带黄": {"带下色黄"},
    "不孕": {"不孕"},
    "怀不上": {"不孕"},

    # ---- 男性 ----
    "阳痿": {"阳痿"},
    "早泄": {"早泄"},
    "遗精": {"遗精"},
    "梦遗": {"梦遗"},
    "滑精": {"滑精"},
    "前列腺": {"淋证", "精浊"},
    "阴囊潮湿": {"阴部潮湿"},

    # ---- 皮肤 ----
    "皮肤痒": {"皮肤瘙痒"},
    "身上痒": {"皮肤瘙痒"},
    "起疹子": {"皮疹", "瘾疹"},
    "长疙瘩": {"皮疹"},
    "皮肤干": {"皮肤干燥"},
    "掉头发": {"脱发"},
    "脱发": {"脱发"},
    "头发白": {"须发早白"},

    # ---- 水肿/体态 ----
    "浮肿": {"水肿"},
    "水肿": {"水肿"},
    "肿": {"水肿"},
    "眼肿": {"目胞浮肿"},
    "脸肿": {"面目浮肿"},
    "脚肿": {"足跗浮肿"},
    "肚子大": {"腹部胀大"},
    "体重增加": {"体重增加"},
    "消瘦": {"消瘦"},
    "瘦": {"消瘦"},
    "胖": {"形体肥胖"},

    # ---- 饮食/口味 ----
    "喜热饮": {"喜喝热水"},
    "喝热水": {"喜喝热水"},
    "喝凉水": {"喜冷饮"},
    "爱喝凉的": {"喜冷饮"},
    "能吃": {"多食"},
    "总饿": {"多食易饥"},
    "吃不胖": {"多食", "消瘦"},

    # ---- 全身 ----
    "身上重": {"身重", "肢体困重"},
    "身体沉": {"身重", "肢体困重"},
    "胳膊沉": {"肢体困重"},
    "腿沉": {"肢体困重"},
    "身上没劲": {"乏力"},
    "出虚汗没劲": {"自汗", "乏力"},
    "怕冷又怕热": {"恶寒发热"},
    "感冒": {"恶寒", "发热", "鼻塞", "流涕"},
    "上火": {"口干", "口苦", "咽喉肿痛"},
}


class TcmMatcher:
    """中医辨病辨证匹配器

    使用方式:
        await tcm_matcher.load(engine)     # 启动时加载
        diseases = tcm_matcher.match_diseases("发热恶寒头痛")  # 匹配疾病
        ctx = tcm_matcher.format_context("发热恶寒头痛")       # 格式化 LLM 上下文
    """

    def __init__(self):
        self._diseases: list[dict[str, Any]] = []
        self._syndromes: list[dict[str, Any]] = []
        self._loaded = False

    async def load(self, engine: AsyncEngine) -> None:
        """从 MySQL 加载疾病和证型数据到内存（表与当前数据库同库）"""
        if not engine:
            logger.warning("TcmMatcher: engine 未就绪，跳过加载")
            return

        try:
            async with engine.connect() as conn:
                # 加载疾病
                disease_sql = text(
                    "SELECT * FROM `tcm_disease` "
                    "WHERE status = 1 ORDER BY sort_order"
                )
                result = await conn.execute(disease_sql)
                rows = result.fetchall()
                self._diseases = [dict(row._mapping) for row in rows]
                logger.info("TcmMatcher: 加载 %d 条疾病数据", len(self._diseases))

                # 加载证型
                syndrome_sql = text(
                    "SELECT * FROM `tcm_syndrome` "
                    "WHERE status = 1 ORDER BY sort_order"
                )
                result = await conn.execute(syndrome_sql)
                rows = result.fetchall()
                self._syndromes = [dict(row._mapping) for row in rows]
                logger.info("TcmMatcher: 加载 %d 条证型数据", len(self._syndromes))

            self._loaded = True
        except Exception as e:
            logger.warning("TcmMatcher: 数据加载失败（服务将继续运行）: %s", e)
            self._diseases = []
            self._syndromes = []
            self._loaded = False

    def get_recommended_formula(self, syndrome_name: str) -> str | None:
        """根据证型名称获取推荐方剂

        Args:
            syndrome_name: 证型名称（如"肾阳虚证"）

        Returns:
            推荐方剂名称（如"右归丸加减"），未找到返回 None
        """
        if not self._loaded or not syndrome_name:
            return None
        name = syndrome_name.strip()
        for s in self._syndromes:
            if s.get("syndrome_name", "").strip() == name:
                return s.get("recommended_formula") or None
        return None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def disease_count(self) -> int:
        return len(self._diseases)

    @property
    def syndrome_count(self) -> int:
        return len(self._syndromes)

    # ----------------------------------------------------------------
    # 匹配方法
    # ----------------------------------------------------------------

    def match_diseases(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """根据症状文本匹配疾病，按匹配度排序

        Args:
            query: 患者的症状描述（主诉、问诊信息等）
            top_k: 返回前 N 个结果

        Returns:
            匹配的疾病列表，每个包含 id/name/description/common_symptoms 等字段
        """
        if not self._loaded or not self._diseases or not query.strip():
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for d in self._diseases:
            score = 0.0
            # 常见症状匹配
            common = d.get("common_symptoms") or ""
            if common:
                score += self._score_match(tokens, common)
            # 主要特征加权
            features = d.get("main_features") or ""
            if features:
                score += self._score_match(tokens, features) * 1.5
            # 疾病描述辅助匹配
            desc = d.get("disease_description") or ""
            if desc:
                score += self._score_match(tokens, desc) * 0.5
            if score > 0:
                scored.append((d, score))

        scored.sort(key=lambda x: -x[1])
        return [self._format_disease(d, s) for d, s in scored[:top_k]]

    def match_syndromes(
        self,
        query: str,
        disease_id: int | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """根据症状文本匹配证型，可按 disease_id 过滤

        Args:
            query: 患者的症状描述
            disease_id: 可选，限定匹配某疾病的证型
            top_k: 返回前 N 个结果

        Returns:
            匹配的证型列表，每个包含 name/type/main_symptoms/tongue_pulse 等字段
        """
        if not self._loaded or not self._syndromes or not query.strip():
            return []

        candidates = self._syndromes
        if disease_id is not None:
            candidates = [s for s in candidates if s.get("disease_id") == disease_id]

        if not candidates:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for s in candidates:
            score = 0.0
            # 主要症状权重高
            main = s.get("main_symptoms") or ""
            if main:
                score += self._score_match(tokens, main) * 2.0
            # 次要症状
            secondary = s.get("secondary_symptoms") or ""
            if secondary:
                score += self._score_match(tokens, secondary)
            # 舌脉辅助
            tp = s.get("tongue_pulse") or ""
            if tp:
                score += self._score_match(tokens, tp) * 0.5
            if score > 0:
                scored.append((s, score))

        scored.sort(key=lambda x: -x[1])
        return [self._format_syndrome(s, sc) for s, sc in scored[:top_k]]

    # ----------------------------------------------------------------
    # 格式化
    # ----------------------------------------------------------------

    def format_context(self, query: str, top_k: int = 3) -> str:
        """格式化为 LLM 上下文字段

        将匹配到的疾病和证型拼接成一段结构化的 Markdown 文本，
        供 LLM 在辨病辨证时参考。

        Args:
            query: 患者症状描述
            top_k: 每种返回前 N 个

        Returns:
            格式化的上下文文本，无匹配时返回空字符串
        """
        diseases = self.match_diseases(query, top_k=top_k)
        syndromes = self.match_syndromes(query, top_k=top_k)

        if not diseases and not syndromes:
            return ""

        parts = [
            "## 辨病辨证参考（结构化数据）",
            "> 以下是根据患者症状关键词匹配的疾病和证型参考，供辨证参考。",
        ]

        if diseases:
            parts.append("\n### 可能疾病（按匹配度排序）")
            for i, d in enumerate(diseases, 1):
                parts.append(f"**{i}. {d['name']}**")
                if d.get("description"):
                    parts.append(f"   - 描述: {d['description']}")
                if d.get("common_symptoms"):
                    parts.append(f"   - 常见症状: {d['common_symptoms']}")
                if d.get("cause_analysis"):
                    parts.append(f"   - 病因: {d['cause_analysis']}")

        if syndromes:
            parts.append("\n### 可能证型（按匹配度排序）")
            for i, s in enumerate(syndromes, 1):
                parts.append(f"**{i}. {s['name']}**（{s['type']}）")
                if s.get("main_symptoms"):
                    parts.append(f"   - 主要表现: {s['main_symptoms']}")
                if s.get("secondary_symptoms"):
                    parts.append(f"   - 次要表现: {s['secondary_symptoms']}")
                if s.get("tongue_pulse"):
                    parts.append(f"   - 舌脉特征: {s['tongue_pulse']}")
                if s.get("treatment_principle"):
                    parts.append(f"   - 治法: {s['treatment_principle']}")
                if s.get("recommended_formula"):
                    parts.append(f"   - 推荐方剂: {s['recommended_formula']}")

        parts.append(
            "\n> **注意**: 以上仅供辨证参考，最终诊断需综合四诊信息确认。"
        )

        return "\n".join(parts)

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """将症状文本切分为关键词 Token，同时做同义词扩展

        按中英文逗号、分号、顿号、句号、空格分割，
        过滤过短的无效 token。对每个 token 查找同义词映射，
        将口语化表述扩展为数据库中的标准中医术语。

        如：患者说"发烧" → 生成 {"发烧", "发热"}，提高对数据库"发热"的命中率。
        """
        if not text:
            return set()
        tokens = re.split(r"[,;，；、。\s]+", text)
        result: set[str] = set()
        for t in tokens:
            t = t.strip()
            if len(t) >= _MIN_KEYWORD_LENGTH:
                result.add(t)
                # 同义词扩展
                if t in _SYMPTOM_SYNONYMS:
                    result.update(_SYMPTOM_SYNONYMS[t])
        return result

    @staticmethod
    def _score_match(tokens: set[str], field_text: str) -> float:
        """计算一组 Token 与症状字段文本的匹配得分

        使用子串匹配（token in field_text），比精确分割匹配更宽容。
        中医症状描述中"发热"可能在字段中以"发热、恶寒"形式出现。
        """
        if not tokens or not field_text:
            return 0.0
        matches = sum(1 for t in tokens if t in field_text)
        return float(matches)

    @staticmethod
    def _format_disease(d: dict, score: float) -> dict[str, Any]:
        return {
            "id": d.get("id"),
            "name": d.get("disease_name", ""),
            "category": d.get("disease_category", ""),
            "description": d.get("disease_description", ""),
            "common_symptoms": d.get("common_symptoms", ""),
            "main_features": d.get("main_features", ""),
            "cause_analysis": d.get("cause_analysis", ""),
            "prognosis": d.get("prognosis", ""),
            "score": round(score, 1),
        }

    @staticmethod
    def _format_syndrome(s: dict, score: float) -> dict[str, Any]:
        return {
            "id": s.get("id"),
            "disease_id": s.get("disease_id"),
            "name": s.get("syndrome_name", ""),
            "type": s.get("syndrome_type", ""),
            "main_symptoms": s.get("main_symptoms", ""),
            "secondary_symptoms": s.get("secondary_symptoms", ""),
            "tongue_pulse": s.get("tongue_pulse", ""),
            "pathogenesis": s.get("pathogenesis", ""),
            "treatment_principle": s.get("treatment_principle", ""),
            "recommended_formula": s.get("recommended_formula", ""),
            "dietary_advice": s.get("dietary_advice", ""),
            "daily_regimen": s.get("daily_regimen", ""),
            "score": round(score, 1),
        }


# 全局单例
tcm_matcher = TcmMatcher()
