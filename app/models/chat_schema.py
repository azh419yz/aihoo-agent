"""对话相关请求/响应 Schema"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# 请求模型
# ============================================================


class HosSickInfo(BaseModel):
    """就诊人信息（医院后端选择的）"""

    name: str = Field(..., description="就诊人姓名")
    gender: str = Field(..., description="性别: male / female")
    age: int = Field(..., description="年龄")
    allergy_history: list[str] = Field(default_factory=list, description="过敏史")
    past_medical_history: list[str] = Field(default_factory=list, description="既往史")


class ChatRequest(BaseModel):
    """对话请求"""

    session_id: str = Field(..., description="问诊会话唯一标识 (UUID)")
    patient_id: str = Field(..., description="患者唯一标识 (UUID)")
    message: str = Field(..., description="用户消息内容")
    medical_record_urls: list[str] | None = Field(default=None, description="病历图片 URL 列表")
    tongue_urls: list[str] | None = Field(default=None, description="舌照图片 URL 列表（UPLOADING_IMAGES 阶段使用）")
    face_urls: list[str] | None = Field(default=None, description="面照图片 URL 列表（UPLOADING_IMAGES 阶段使用）")
    paid: bool = Field(..., description="是否已付费")
    hos_sick_info: HosSickInfo | None = Field(default=None, description="后端选择的就诊人信息")
    action: str = Field(default="CHAT", description="操作指令")


class CreateSessionRequest(BaseModel):
    """新建会话请求"""

    session_id: str = Field(..., description="问诊会话唯一标识 (UUID)")
    patient_id: str = Field(..., description="患者唯一标识 (UUID)")


# ============================================================
# 响应模型
# ============================================================


class InquiryJson(BaseModel):
    """问诊信息"""

    symptoms: list[str] = Field(default_factory=list, description="症状列表")
    duration: str = Field(default="", description="持续时间")
    accompanying_symptoms: list[str] = Field(default_factory=list, description="伴随症状")
    # 系统问诊各维度文本（付费后采集）
    sleep: str = Field(default="", description="睡眠情况")
    diet: str = Field(default="", description="饮食情况")
    stool: str = Field(default="", description="大便情况")
    urine: str = Field(default="", description="小便/男科局部情况")
    emotion: str = Field(default="", description="情志情况")
    thermo: str = Field(default="", description="寒热情况")
    tongue: str = Field(default="", description="舌象")
    face: str = Field(default="", description="面色")
    pulse: str = Field(default="", description="脉象")


class DrugItem(BaseModel):
    """药品"""

    name: str = Field(..., description="药品名称")
    number: str = Field(default="", description="数量克数")


class InstructionJson(BaseModel):
    """用法信息"""

    usage: str = Field(default="1", description="用法: 1=内服 2=外用")
    doseNumber: str = Field(default="", description="全部剂数")
    dose: str = Field(default="", description="每日剂量")
    times: str = Field(default="", description="每剂使用次数")
    decoctionSize: str = Field(default="1", description="煎药规格: 1=100ml/袋 2=200ml/袋")
    advice: str = Field(default="", description="医嘱")
    remark: str = Field(default="", description="备注")


class PrescriptionJson(BaseModel):
    """处方信息（新格式）"""

    disease: str = Field(default="", description="疾病诊断")
    syndrome: str = Field(default="", description="证型（多个用逗号分隔）")
    drugList: list[DrugItem] = Field(default_factory=list, description="药品列表")
    instruction: InstructionJson | None = Field(default=None, description="用法信息")


class DiagnosisJson(BaseModel):
    """辨病辨证结果"""

    disease: str = Field(default="", description="疾病诊断")
    syndrome: str = Field(default="", description="证型")


class ResponseData(BaseModel):
    """响应业务数据"""

    chief_complaint: str | None = Field(default=None, description="主诉")
    inquiry_json: InquiryJson | None = Field(default=None, description="问诊信息")
    diagnosis_json: DiagnosisJson | None = Field(default=None, description="辨病辨证结果")
    prescription_json: PrescriptionJson | None = Field(default=None, description="处方信息")
    patient_mismatch: bool = Field(default=False, description="患者信息是否不匹配（仅 SELECT_PATIENT 阶段有业务意义）")
    mismatch_reason: str | None = Field(default=None, description="不匹配原因")
    need_medical_record: bool = Field(default=False, description="是否需要上传线下病历（引导上传病历/处方时启用）")
    need_upload_image: bool = Field(default=False, description="是否需要上传舌面照（引导上传舌照/面照时启用）")
    need_pay: bool = Field(default=False, description="是否需要付费（引导付费时启用）")
    need_select: bool = Field(default=False, description="是否需要重新选择就诊人")
    diagnosis_done: bool = Field(default=False, description="辨证是否已完成（true 时表示可向 agent 请求开方）")


class ChatResponse(BaseModel):
    """对话响应"""

    session_id: str = Field(..., description="问诊会话唯一标识")
    message: str = Field(..., description="Agent 回复内容")
    action: str = Field(..., description="业务动作标识")
    meta: ResponseData | None = Field(default=None, description="业务数据（附加业务上下文）")
