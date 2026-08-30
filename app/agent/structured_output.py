"""结构化输出 Schema 定义

用于 orchestrator.ainvoke_structured 的 json_schema 结构化输出。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PatientConfirmAction(str, Enum):
    """用户对就诊人确认的意图分类"""
    CONFIRM = "confirm"  # 确认就诊人信息正确
    DISAGREE = "disagree"  # 不同意/重新选择
    UNKNOWN = "unknown"  # 表达不明确


class ExtractedPatientInfo(BaseModel):
    """从对话中提取的患者基本信息"""

    name: str | None = Field(default=None, description="姓名")
    gender: str | None = Field(default=None, description="性别: male/female")
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
