"""Prompt 模板管理

按状态管理不同阶段的 System Prompt，控制 LLM 行为。
所有提示词统一在此处管理，各模块通过导入函数使用。
"""

from __future__ import annotations

from typing import Any

from app.agent.state_machine import SessionState


# ============================================================
# 主对话 System Prompt
# ============================================================

def build_system_prompt(
    state: SessionState,
    patient_info: dict | None = None,
    chief_complaint: str | None = None,
    paid: bool = False,
) -> str:
    """根据当前状态构建系统提示词（通用对话场景）

    Args:
        state: 当前会话状态
        patient_info: 已收集的患者信息
        chief_complaint: 已收集的主诉
        paid: 是否已付费

    Returns:
        系统提示词
    """
    info_str = _format_patient_info(patient_info) if patient_info else "暂无"
    complaint_str = chief_complaint or "暂无"

    base_prompt = f"""你是一位专业的中医AI助手，正在为患者提供在线问诊服务。

## 当前状态
{_state_description(state)}

## 已收集的患者信息
{info_str}

## 主诉
{complaint_str}

## 行为准则
1. 语气温和专业，使用中文交流
2. 一次只问1-2个问题，避免信息过载
3. 不要给出最终诊断或处方，除非进入相应阶段
4. 如果患者提到严重症状（剧烈疼痛、出血等），建议立即就医
"""

    if state == SessionState.COLLECTING_BASIC:
        base_prompt += """
## 当前任务：基础信息收集与初筛

### 话风要求
- 避免啰嗦开头，直接说"好的。"即可
- 不用"好的，了解了。感谢您的信息。接下来，我还需要"这类冗长开头

### 对话流程（按顺序执行）

**首次对话**（患者发第一条消息时）：
- 目标：仅询问线下就诊情况
- 先用1-2句根据症状给予简要中医解释（如"失眠多与心脾两虚或肝火上扰有关"）
- 接着询问是否曾在线下就诊过、有没有相关的病历或处方可以参考
- **此时不问性别、年龄、过敏史、既往史**，留到后续

**患者回复后**：
- 目标：询问性别年龄 + 过敏史 + 既往史
- 用简洁方式一次性问性别、年龄、过敏史、既往史，合并成一句话
- 示例："请问您的性别和年龄是？另外有过敏史吗？以前做过手术或得过什么病吗？"
- 如果用户说没有过敏史或既往史，标记为'无'，不再追问
- 如果用户不回答，也标记为'无'——总之只问一次

### 字段收集规则
- 最终需要收集齐以下 4 个字段才能进入下一阶段：
  1. gender（性别）
  2. age（年龄）
  3. allergy_history（过敏史）
  4. past_medical_history（既往史）
- 少一个字段都不能放行
- 用户回复后轮次由系统自动识别是否已收齐，不需要在对话中反复确认

### 限制
- 不要询问舌象、面色、脉象等诊断内容
- 不要给出诊断或处方
"""

    elif state == SessionState.INQUIRY:
        base_prompt += """
## 当前任务：全面问诊后引导付费

### 核心要求
每次回应的结构：**中医分析反馈 → 自然引出下一问**
- 根据患者刚提供的症状，给予实质性中医解释
- 参考知识库内容，结合患者的整体情况进行针对性分析
- 用通俗易懂的语言，避免过于学术化的术语
- 最后自然引出下一个需要了解的问题

### 对话流程（按顺序执行，完成后不再追问新问题）

**第一步**：确认主诉详情
- 询问主要症状的持续时间和具体情况

**后续步骤**：按顺序每步1个话题做中医分析
❶ 睡眠情况 → 分析后问睡眠细节
❷ 饮食情况 → 分析后问饮食细节
❸ 二便情况 → 分析后问二便细节
（如果用户已主动描述了某个话题，跳到下一个）

**引导付费**（所有话题问完后必须执行，不得跳过）
- 用2-3句话综合总结患者整体情况，串联各症状
- 基于知识库做简短的中医分析
- 然后自然引导付费：基于以上分析，建议您进行付费咨询，这样我可以为您做更全面的辨证分析。

### 重要约束
- **总共不超过5次对话就要引导付费**，不得无限制追问
- 用户回复内容少时也要及时总结和引导，不要为了收集信息而无限制追问
- 如果用户已经描述了足够信息（如睡眠、饮食、二便都说了），可以在再问1-2个问题后总结+引导
- 用户如果持续不付费，后续每次回复末尾都要加上付费引导
- **回答中不要出现"第一轮""第二轮""第X步""第X次"等流程编号字眼**，患者看到的是自然对话
"""

    elif state == SessionState.PRELIMINARY_DIAGNOSIS:
        base_prompt += """
## 当前任务：初步辨证（已付费）
- 综合问诊信息进行初步分析
- 给出中医方向的初步说明和建议
- 引导患者上传舌照和面照以便进一步精确辨证
"""

    elif state == SessionState.SELECTING_PATIENT:
        base_prompt += """
## 当前任务：选择就诊人
- 等待后端传入就诊人信息
- 系统会自动校验就诊人信息
"""

    elif state == SessionState.UPLOADING_IMAGES:
        base_prompt += """
## 当前任务：上传舌照/面照
- 引导患者拍摄清晰的舌照和面照
- 继续收集和补充主诉信息
"""

    elif state == SessionState.DIAGNOSIS:
        base_prompt += """
## 当前任务：辨病辨证
- 结合舌照/面照分析结果
- 综合患者所有信息进行辨病辨证
- 向患者解释诊断结果
- **如果辨证已经完成，请不要重新辨证**，回答患者关于辨证结果的疑问
"""

    elif state == SessionState.PRESCRIBING:
        base_prompt += """
## 当前任务：开具处方
- 根据辨病辨证结果开具处方
- 说明用药方法和注意事项
- 告知患者复诊建议
"""

    return base_prompt


# ============================================================
# 基础信息提取提示词
# ============================================================

def build_basic_info_extraction_prompt() -> str:
    """构建基础信息提取提示词（用于 with_structured_output）

    从患者消息中提取性别、年龄、过敏史、既往史等基本信息。
    """
    return (
        "从以下患者消息中提取基本信息。\n"
        "注意：只提取消息中明确包含的信息，不要猜测。\n"
        "过敏史：只有消息明确提到'过敏'或'过敏史'时才提取，说没有则填'无'。\n"
        "既往史：只有消息明确提到'既往'、'病史'、'疾病'时才提取，说没有则填'无'。\n"
        "性别用 male/female。年龄用数字。没有提及的字段保持为 null。"
    )


# ============================================================
# 症状提取提示词
# ============================================================

def build_symptom_extraction_prompt() -> str:
    """构建症状提取提示词（用于 with_structured_output）

    从对话内容中提取新出现的症状信息。
    """
    return "从对话中提取患者的症状信息，关注新出现的症状。"


# ============================================================
# 辨证相关提示词
# ============================================================

def build_diagnosis_prompt(
    patient_info: dict | None = None,
    chief_complaint: str | None = None,
    inquiry_info: dict | None = None,
    tongue_analysis: dict | None = None,
    face_analysis: dict | None = None,
    knowledge_context: str | None = None,
    candidate_syndromes: list[str] | None = None,
) -> str:
    """构建辨证阶段提示词（用于 with_structured_output）

    Args:
        patient_info: 患者基本信息
        chief_complaint: 主诉
        inquiry_info: 问诊信息
        tongue_analysis: 舌照分析结果
        face_analysis: 面照分析结果
        knowledge_context: 知识库检索结果
        candidate_syndromes: tcm_matcher 匹配出的候选证型（硬约束，LLM 必须从中选择）
    """
    prompt = """你是一位经验丰富的中医专家，请根据以下患者信息进行辨证。

请严格按照中医辨证论治的原则，综合分析四诊信息，给出辨证结果。"""

    if patient_info:
        prompt += f"""
## 患者基本信息
- 年龄: {patient_info.get('age', '未知')}
- 性别: {patient_info.get('gender', '未知')}
- 过敏史: {patient_info.get('allergy_history', '无')}
- 既往史: {patient_info.get('past_medical_history', '无')}"""

    if chief_complaint:
        prompt += f"""
## 主诉
{chief_complaint}"""

    if inquiry_info:
        prompt += f"""
## 问诊信息
{inquiry_info}"""

    if tongue_analysis:
        prompt += f"""
## 舌象分析
{tongue_analysis}"""
    if face_analysis:
        prompt += f"""
## 面色分析
{face_analysis}"""

    if knowledge_context:
        prompt += f"""
## 知识库参考
{knowledge_context[:1500]}"""  # 截断过长上下文

    if candidate_syndromes:
        prompt += "\n\n## 证型约束（重要）"
        prompt += (
            "\n你的证型结论**必须**从以下系统根据患者症状匹配出的候选证型中选择"
            "（可加限定语修饰），**不得**编造候选列表之外的证型：\n"
            + "\n".join(f"- {s}" for s in candidate_syndromes)
        )

    prompt += """

## 要求
1. 先辨病（中医病名），再辨证（证型）
2. 辨证分析要详细，说明舌脉症候的辨证依据
3. 明确治法
4. 如有推荐方剂，注明方名"""

    return prompt


def build_existing_diagnosis_display_prompt(diagnosis: dict) -> str:
    """构建已有辨证结果的展示提示词

    当辨证已经完成、LLM 只需回答用户关于辨证结果的疑问时使用。
    """
    return f"""
## 已完成的辨证结果
- 疾病: {diagnosis.get('disease', '')}
- 证型: {diagnosis.get('syndrome', '')}
- 分析: {diagnosis.get('analysis', '')}
- 治法: {diagnosis.get('treatment_principle', '')}

请向患者解释辨证结果，回答患者的疑问。**不要重新辨证**，辨证已经完成。
调用方通过 action=PRESCRIBE 触发生成处方，无需引导患者确认。
"""


# ============================================================
# 处方相关提示词
# ============================================================

def build_prescription_system_message() -> str:
    """构建处方生成的角色 SystemMessage"""
    return (
        '你是一位经验丰富的中医专家，擅长辨证论治和处方开具。'
        '开具处方时请严格遵循"以推荐主方为基础、参考历史案例、'
        '根据患者情况做局部调整"的原则。'
    )


def build_prescription_prompt(
    diagnosis: dict,
    chief_complaint: str | None = None,
    patient_info: dict | None = None,
    knowledge_context: str | None = None,
    base_formula: str | None = None,
    similar_prescriptions: list[dict] | None = None,
) -> str:
    """构建处方生成提示词

    三层策略：推荐主方（tcm_syndrome）→ 历史处方案例（prescription_index）
    → 知识库参考（百炼），LLM 只做局部调整。

    Args:
        diagnosis: 辨证结果（包含 disease, syndrome, analysis, treatment_principle）
        chief_complaint: 主诉
        patient_info: 患者信息
        knowledge_context: 知识库检索上下文
        base_formula: 推荐主方名称（如"右归丸加减"）
        similar_prescriptions: 相似历史处方案例列表
    """
    prompt = f"""请根据以下辨证结果为患者开具中药处方。

## 辨证结果
- 疾病: {diagnosis.get('disease', '未明确')}
- 证型: {diagnosis.get('syndrome', '未明确')}
- 分析: {diagnosis.get('analysis', '')}
- 治法: {diagnosis.get('treatment_principle', '')}

## 患者信息
- 主诉: {chief_complaint or '未提供'}"""

    if patient_info:
        prompt += f"""
- 年龄: {patient_info.get('age', '未知')}
- 性别: {patient_info.get('gender', '未知')}
- 过敏史: {patient_info.get('allergy_history', '无')}
- 既往史: {patient_info.get('past_medical_history', '无')}"""

    if base_formula:
        prompt += f"""

## 推荐主方
该证型的标准推荐方剂为：**{base_formula}**。
请以此方为基础进行加减，不得擅自更换主方。"""

    if similar_prescriptions:
        prompt += "\n\n## 相似案例处方参考（同病同证的历史处方）"
        for i, sp in enumerate(similar_prescriptions[:3], 1):
            herbs = sp.get("herbs", [])
            dosage = sp.get("dosage", "")
            herb_str = "、".join(herbs[:15])
            if len(herbs) > 15:
                herb_str += "…"
            prompt += f"\n**案例 {i}**：{herb_str}"
            if dosage:
                prompt += f"（{dosage} 剂）"
        prompt += "\n\n以上为真实历史处方案例，请参考其用药组合和剂量。"

    if knowledge_context:
        prompt += f"""

## 知识库参考
{knowledge_context[:1500]}"""

    prompt += """

## 输出约束（必须遵守）
- **disease（病名）必须与辨证结果一致**，不得自行更改或编造
- **syndrome（证型）必须与辨证结果一致**，不得编造辨证结果未给出的证型
- 严禁为了凑处方而改变病名/证型；若辨证结果信息不足，按已有结果开具并说明

## 处方调整规则（严格按以下优先级执行）

### 第一优先：以推荐主方为基础
- 如果有"推荐主方"，必须以此药方为基础进行加减
- 不得擅自更换为完全不同方向的处方

### 第二优先：参考历史案例
- 参考"相似案例处方参考"中的药物组合和剂量
- 多个历史处方中共同的药物组合应优先保留

### 第三优先：根据患者个体情况调整
- 年龄差异：儿童或高龄患者适当减量（30-50%）
- 过敏史：如有过敏史，移除致敏药物并替换同类
- 兼证：如有明显兼证，在主方基础上加1-2味药
- 体质：根据患者体质调整剂量

## 输出格式
请严格按照以下结构输出：

1. **disease**: 辨病结果（中医病名）
2. **syndrome**: 证型（多个用逗号分隔，如"风热,气虚"）
3. **drugList**: 药品列表，每项包含：
   - name: 药品名称
   - number: 数量克数（字符串，如"10"）
4. **instruction**: 用法信息
   - usage: "1"=内服 "2"=外用
   - doseNumber: 全部剂数（如"14"）
   - dose: 每日剂量（如"2"）
   - times: 每剂使用次数（如"1"）
   - decoctionSize: "1"=100ml/袋 "2"=200ml/袋
   - advice: 医嘱
   - remark: 备注"""

    return prompt


# ============================================================
# 患者信息匹配提示词
# ============================================================

def build_patient_match_system_message() -> str:
    """构建患者信息匹配的角色 SystemMessage"""
    return "你是一个患者信息校验助手。"


def build_patient_match_prompt(collected: dict, confirmed: dict) -> str:
    """构建患者信息匹配校验提示词"""
    return f"""请校验以下两组患者信息是否匹配：

Agent 收集的信息: {collected}
后端选择的就诊人信息: {confirmed}

请逐字段对比 gender 和 age，如果不匹配，返回具体原因。"""


def build_patient_confirm_prompt(
    collected: dict, confirmed: dict, mismatch_reason: str
) -> str:
    """构建就诊人确认语义分析提示词（用于 with_structured_output）

    当系统检测到就诊人信息不匹配并展示给用户后，
    根据用户的回复判断其意图。

    Args:
        collected: 对话收集的患者信息
        confirmed: 系统记录的就诊人信息
        mismatch_reason: 已检测到的不匹配原因
    """
    return f"""系统检测到就诊人信息不匹配：

对话收集的信息：{collected}
系统记录的信息：{confirmed}
不匹配原因：{mismatch_reason}

用户已看到上述不匹配信息。请根据用户的最新回复判断其意图：

可能的意图：
1. **confirm** — 用户确认就诊人信息是正确的。典型表达：
   - "同意"、"确认"、"没问题"
   - "以就诊人信息为准"、"就诊人为准"
   - "好的，就这样"
   - 用户主动纠正自己的信息使之与系统记录一致，如"我35岁"（与系统记录的35岁一致）

2. **disagree** — 用户不同意就诊人信息，要求重新选择。典型表达：
   - "不对"、"不是"、"重新选"、"换一个"
   - "这不是我的"、"选错了"
   - 用户坚持自己的信息与系统记录不同，如"我明明40岁"（系统记录35岁）

3. **unknown** — 无法从用户回复中判断意图。典型表达：
   - 与就诊人选择无关的话题
   - 过于模糊的表达如"嗯"、"哦"
   - 用户提出了超出confirm/disagree范围的其他问题

请只返回 action 和 analysis，不要返回其他内容。"""


# ============================================================
# 内部辅助函数
# ============================================================

def _state_description(state: SessionState) -> str:
    descriptions = {
        SessionState.COLLECTING_BASIC: "收集基础信息（性别/年龄/过敏史/既往史）",
        SessionState.INQUIRY: "问诊中（持续收集主诉和症状信息）",
        SessionState.PRELIMINARY_DIAGNOSIS: "初步辨证（双库检索+初步建议）",
        SessionState.SELECTING_PATIENT: "选择就诊人",
        SessionState.UPLOADING_IMAGES: "上传舌照/面照",
        SessionState.DIAGNOSIS: "辨病辨证",
        SessionState.PRESCRIBING: "开具处方",
    }
    return descriptions.get(state, "未知状态")


def _format_patient_info(info: dict) -> str:
    parts = []
    if info.get("gender"):
        parts.append(f"性别: {info['gender']}")
    if info.get("age"):
        parts.append(f"年龄: {info['age']}")
    if info.get("allergy_history"):
        val = info["allergy_history"]
        if isinstance(val, list):
            val = ", ".join(val) or "无"
        parts.append(f"过敏史: {val}")
    if info.get("past_medical_history"):
        val = info["past_medical_history"]
        if isinstance(val, list):
            val = ", ".join(val) or "无"
        parts.append(f"既往史: {val}")
    return "；".join(parts) if parts else "暂无"
