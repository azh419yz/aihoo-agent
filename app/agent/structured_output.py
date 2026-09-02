"""结构化输出 Schema 定义

用于 orchestrator.ainvoke_structured 的 json_schema 结构化输出。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.chat_schema import QuestionChoice


class PatientConfirmAction(str, Enum):
    """用户对就诊人确认的意图分类"""
    CONFIRM = "confirm"  # 确认就诊人信息正确
    DISAGREE = "disagree"  # 不同意/重新选择
    UNKNOWN = "unknown"  # 表达不明确


class ExtractedPatientInfo(BaseModel):
    """从对话中提取的患者基本信息"""

    name: str | None = Field(default=None, description="姓名")
    gender: str | None = Field(default=None, description="性别: 男/女")
    age: int | None = Field(default=None, description="年龄")
    height: str | None = Field(default=None, description="身高（如'175cm'）")
    occupation: str | None = Field(default=None, description="职业")
    weight: str | None = Field(default=None, description="体重（如'70kg'）")
    allergy_history: str | None = Field(default=None, description="过敏史，患者说没有填'无'，否则填具体过敏原")
    past_medical_history: str | None = Field(default=None, description="既往史，患者说没有填'无'，否则填具体疾病")
    chief_complaint: str | None = Field(default=None, description="主诉")
    has_enough_info: bool = Field(default=False, description="信息是否足够进入下一阶段")


class PatientMatchResult(BaseModel):
    """患者信息匹配结果"""

    is_match: bool = Field(..., description="是否匹配")
    mismatches: list[str] = Field(default_factory=list, description="不匹配的字段列表")
    reason: str = Field(default="", description="详细原因")


class SymptomExtraction(BaseModel):
    """从问诊中提取的症状信息 + 问诊维度进度"""

    symptoms: list[str] = Field(default_factory=list, description="已确认的症状列表")
    new_symptoms: list[str] = Field(default_factory=list, description="本轮新发现的症状")
    duration: str | None = Field(default=None, description="病程/持续时间")
    accompanying_symptoms: list[str] = Field(default_factory=list, description="伴随症状")
    key_findings: str = Field(default="", description="关键发现")

    # ---- 问诊进度追踪 ----
    covered_dimensions: list[str] = Field(
        default_factory=list,
        description="本轮已覆盖的问诊维度，取值：chief_complaint/sleep/diet/stool/urine/emotion/thermo",
    )
    dimension_findings: dict[str, str] = Field(
        default_factory=dict,
        description="各已覆盖维度的文本总结，如 {'sleep': '入睡难，多梦'}",
    )
    chief_complaint_done: bool = Field(
        default=False, description="付费前主诉链路是否已问清（问清后应引导付费）"
    )

    need_more_info: bool = Field(default=True, description="是否还需要更多信息")
    next_question: str = Field(default="", description="下一个应该问的问题方向")


class SupplementExtraction(BaseModel):
    """补充信息意图判断（辨证前最后一步补充确认）

    判断用户回复是否提供了真正需要补充的信息；没有补充（"没有了"等）时 has_supplement=false。
    """

    has_supplement: bool = Field(
        default=False,
        description="用户是否补充了新的信息（true=有补充；false=明确表示没有补充，如'没有了'）",
    )
    new_symptoms: list[str] = Field(
        default_factory=list, description="补充信息中明确提及的新症状/异常点"
    )


class QuestionChoices(BaseModel):
    """助手回复中的问答选项（专用结构化输出，供前端渲染可点选 chips）

    每条 choices = 一个独立的**封闭式选择题**（是/否、有/没有、A还是B、
    是A、B还是C、上传/不上传等，或明确列出备选）；回复在一个自然段里问了
    多个封闭式问题 → 每个问题一条，不要合并、不要遗漏。
    开放/主观问题（性别、年龄、症状描述等无备选答案）→ choices=[]。
    """

    choices: list[QuestionChoice] = Field(
        default_factory=list,
        description=(
            "问答选项块列表；每条含 title（项目/主题）、type（null/single/multi）、"
            "options（选项文本，要完整覆盖回复中的全部备选、简短、用回复原文、"
            "总数不超过 4 个以 3 个为佳；同一回复要么全部选项化、要么全部为空）"
        ),
    )


class MaleInquirySufficiency(BaseModel):
    """男科针对性追问的信息充分度判定（达到最低轮次后，每轮判断能否转辨证）"""

    sufficient: bool = Field(
        default=False,
        description="当前已收集的男科症状信息是否足以确认辨证结果并据此开方",
    )
    missing_areas: list[str] = Field(
        default_factory=list, description="仍缺失、需要继续追问确认的关键方面"
    )


class TongueAnalysisResult(BaseModel):
    """舌诊分析结果（增强版，含证型提示）"""

    tongue_color: str = Field(default="", description="舌色（淡红/红/绛/紫/淡白）")
    tongue_shape: str = Field(default="", description="舌形（胖大/瘦薄/齿痕/裂纹）")
    coating_color: str = Field(default="", description="苔色（白/黄/灰/黑）")
    coating_texture: str = Field(default="", description="苔质（薄/厚/腻/燥/滑）")
    analysis: str = Field(default="", description="舌诊分析")
    syndrome_hints: list[str] = Field(default_factory=list, description="证型提示")


class DiagnosisResult(BaseModel):
    """辨证结果"""

    disease: str = Field(default="", description="疾病诊断（中医病名）")
    syndrome: str = Field(..., description="证型")
    analysis: str = Field(default="", description="辨证分析")
    treatment_principle: str = Field(default="", description="治法")
    formula_name: str | None = Field(default=None, description="推荐方剂名称")


class DrugItem(BaseModel):
    """处方中的药品"""

    name: str = Field(..., description="药品名称")
    number: str = Field(default="", description="数量（克数）")


class InstructionInfo(BaseModel):
    """处方用法信息"""

    usage: str = Field(default="1", description="用法: 1=内服 2=外用")
    doseNumber: str = Field(default="", description="全部剂数")
    dose: str = Field(default="", description="每日剂量（剂）")
    times: str = Field(default="", description="每剂使用次数")
    decoctionSize: str = Field(default="1", description="煎药规格: 1=100ml/袋 2=200ml/袋")
    advice: str = Field(default="", description="医嘱")
    remark: str = Field(default="", description="备注")


class PatientConfirmResult(BaseModel):
    """用户对就诊人确认响应的意图分析

    当系统检测到就诊人信息不匹配时，LLM 根据用户的回复语义
    判断用户是确认、拒绝还是不明确。
    """

    action: PatientConfirmAction = Field(
        default=PatientConfirmAction.UNKNOWN,
        description="确认(confirm)/拒绝(disagree)/不明确(unknown)",
    )
    analysis: str = Field(
        default="",
        description="意图分析理由，如'用户确认就诊人信息正确'/'用户要求重新选择'等",
    )


class PrescriptionResult(BaseModel):
    """处方生成结果"""

    disease: str = Field(default="", description="疾病诊断（辨病结果）")
    syndrome: str = Field(default="", description="证型（多个用逗号分隔）")
    drugList: list[DrugItem] = Field(default_factory=list, description="药品列表")
    instruction: InstructionInfo = Field(default_factory=InstructionInfo, description="用法信息")


class CaseSelectionResult(BaseModel):
    """知识库选案结果：从候选历史案例中选出最匹配的一个

    选中后处方由代码从该案例【处方】原样解析，LLM 不接触药方。
    """

    selected_index: int = Field(
        default=0, description="选中案例序号（1-based，指向候选列表）"
    )
    analysis: str = Field(
        default="", description="选择原因（辨病辨证一致性、主诉吻合、年龄/病史匹配等）"
    )
