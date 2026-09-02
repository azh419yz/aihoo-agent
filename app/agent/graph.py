"""问诊编排状态图（LangGraph）

将 consultation_service 里的 8 个 `_handle_*` 状态处理逻辑逐行平移为 LangGraph 节点。

核心原则：
  - 图只做「一次 chat 请求的执行编排」，不承担持久化。
    会话状态在图的 final state 中按白名单一次性写回 Redis/MySQL（all-or-nothing）。
  - 节点不直接调用 set_field / update_state / sync_to_mysql。
  - 业务逻辑逐行平移，不改变；改的是「编排方式」和「持久化时机」。

字段命名前缀：
  - 会话持久化字段（无前缀）：图成功后按白名单写回
  - `request_*`：本次请求输入，不写回
  - `response_*` / `_deleted_fields`：输出 / 瞬态
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.orchestrator import LLMOrchestrator
from app.agent.prompts import (
    build_basic_info_extraction_prompt,
    build_choices_extraction_prompt,
    build_diagnosis_prompt,
    build_existing_diagnosis_display_prompt,
    build_male_inquiry_sufficiency_prompt,
    build_supplement_extraction_prompt,
    build_symptom_extraction_prompt,
    build_system_prompt,
)
from app.agent.state_machine import STATE_KNOWLEDGE_BASE, ActionType, SessionState
from app.agent.structured_output import (
    DiagnosisResult,
    ExtractedPatientInfo,
    MaleInquirySufficiency,
    QuestionChoices,
    SupplementExtraction,
    SymptomExtraction,
)
from app.agent.tools import prescribe_from_kb, retrieve_knowledge
from app.knowledge.qianfan import retrieve_with_filter
from app.knowledge.tcm_matcher import tcm_matcher
from app.models.chat_schema import InquiryJson, QuestionChoice, ResponseData
from app.multimodal.medical_record import analyze_medical_record, format_medical_record_basic_info
from app.multimodal.tongue_face import analyze_face_images, analyze_tongue_images

logger = logging.getLogger(__name__)


# ============================================================
# State schema
# ============================================================

class ConsultationState(TypedDict, total=False):
    """图内状态。字段分三类，命名前缀防止写回冲突。"""

    # ---- 会话持久化字段（图后按白名单写回 Redis/MySQL）----
    state: str
    paid: bool
    patient_info: dict
    patient_info_confirmed: dict
    chief_complaint: str
    inquiry: dict
    diagnosis: dict
    prescription: dict
    prescription_reason: dict  # 开方选案分析原因（写 Redis 审计，不展示给用户）
    image_urls: list
    tongue_analysis: list
    face_analysis: list
    patient_mismatch: bool
    mismatch_reason: str
    med_record_pending_confirm: bool
    collecting_round: int
    offline_medical_record: dict
    hos_sick_info: dict
    preliminary_diagnosis: dict
    patient_select_pending: bool
    inquiry_progress: dict  # 问诊维度进度（主诉 + 6 个系统维度 → bool）
    # 会话历史（只读，由 save_message 追加，不写回）
    messages: list

    # ---- 本次请求输入（不写回）----
    request_message: str
    request_action: str
    request_paid: bool
    request_hos_sick_info: dict | None
    request_medical_record_urls: list
    request_tongue_urls: list
    request_face_urls: list

    # ---- 输出 / 瞬态 ----
    response_text: str
    response_action: ActionType
    response_data: ResponseData
    _next: str  # 预留：路由指令；当前条件边直接读业务字段，未写入
    _deleted_fields: Annotated[list, operator.add]


# ============================================================
# 通用助手函数（从 consultation_service 迁移）
# ============================================================

async def build_chat_messages(
        system_prompt: str,
        state: dict[str, Any],
        user_message: str,
        session_state: SessionState,
) -> list:
    """构建带知识库上下文和历史对话的 LLM 消息列表

    System Prompt + 知识库上下文 + 历史对话（user/ai交替） + 当前用户消息
    """
    # 按当前状态检索知识库（未付费=通用，付费后=通用+专家）
    kb_type = STATE_KNOWLEDGE_BASE.get(session_state, "general")
    knowledge_context = await retrieve_knowledge(
        query=user_message,
        knowledge_base=kb_type,
        top_k=3,
    )
    if knowledge_context:
        system_prompt = f"{system_prompt}\n\n## 知识库参考\n{knowledge_context}"

    messages: list = [{"role": "system", "content": system_prompt}]

    # 加载历史消息
    history = state.get("messages", [])
    for msg in history[-10:]:  # 最多取最近 10 条
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "ai":
            messages.append({"role": "assistant", "content": content})

    # 当前消息
    messages.append({"role": "user", "content": user_message})
    return messages


def simple_patient_match(collected: dict, confirmed: dict) -> tuple[bool, str]:
    """简单的就诊人信息字段比较（不调用 LLM，避免额外延迟）

    Returns:
        (is_match, reason) — reason 在匹配时为 ""
    """
    reasons = []

    # 性别比较（归一化处理）
    male_set = {"男", "male", "m", "1"}
    female_set = {"女", "female", "f", "2"}

    raw_cg = (collected.get("gender") or "").strip().lower()
    raw_cog = (confirmed.get("gender") or "").strip().lower()
    norm_cg = "male" if raw_cg in male_set else ("female" if raw_cg in female_set else raw_cg)
    norm_cog = "male" if raw_cog in male_set else ("female" if raw_cog in female_set else raw_cog)
    if norm_cg and norm_cog and norm_cg != norm_cog:
        reasons.append(f"性别不一致（对话收集：{collected.get('gender')}，系统记录：{confirmed.get('gender')}）")

    # 年龄比较
    raw_age = str(collected.get("age") or "")
    raw_cage = str(confirmed.get("age") or "")
    if raw_age and raw_cage and raw_age != raw_cage:
        reasons.append(f"年龄不一致（对话收集：{raw_age}岁，系统记录：{raw_cage}岁）")

    return (False, "；".join(reasons)) if reasons else (True, "")


def has_enough_patient_info(patient_info: dict) -> bool:
    """四个字段全部收齐才前进（拒绝空字符串和 None）"""
    return bool(
        patient_info.get("gender")
        and patient_info.get("age")
        and patient_info.get("allergy_history") not in (None, "")
        and patient_info.get("past_medical_history") not in (None, "")
    )


# ============================================================
# 病历 / 基础信息 处理助手
# ============================================================

_REUPLOAD_RECORD_KEYWORDS = ("重新上传", "重新传", "重传", "传错", "重新发", "换一个")


_PHOTO_REUPLOAD_KEYWORDS = ("重新上传", "重新传", "重传", "重新发")


async def _extract_choices(
    orchestrator: LLMOrchestrator, user_msg: str, llm_response: str
) -> list[QuestionChoice]:
    """用专用结构化 model（QuestionChoices）让 LLM 判断并输出问答选项

    返回选项块列表（每条含 title/type/options，多个封闭式问题拆多条）；
    无选项返回 []。失败容错不阻断主流程。
    """
    try:
        result = await orchestrator.ainvoke_structured(
            QuestionChoices,
            [
                {"role": "system", "content": build_choices_extraction_prompt()},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": llm_response},
            ],
        )
        if result and result.choices:
            return [c for c in result.choices if c.options]
    except Exception as e:
        logger.debug("问答选项提取跳过: %s", e)
    return []


async def _set_choices(
    response_data: Any,
    orchestrator: LLMOrchestrator,
    user_msg: str,
    llm_response: str,
) -> None:
    """用 QuestionChoices 结构化提取问答选项并写入 response_data（meta）"""
    choices = await _extract_choices(orchestrator, user_msg, llm_response)
    if choices:
        response_data.options = choices


def _is_reupload_photo_intent(message: str) -> bool:
    """检测用户是否要求重新上传舌面照/照片（与病历重传区分：需含 舌/面/照片）"""
    if not message:
        return False
    if not any(k in message for k in _PHOTO_REUPLOAD_KEYWORDS):
        return False
    return any(k in message for k in ("舌", "面", "照片", "舌面", "舌照", "面照", "图片"))


def _is_reupload_record_intent(message: str) -> bool:
    """检测用户是否表达重新上传病历的意图

    需同时提到"病历/病例"，避免把"重新上传舌照/面照"误判为病历重传
    （舌面照重传由 diagnosis/prescribing 节点内的已有逻辑处理）。
    """
    if not message:
        return False
    if "病历" not in message and "病例" not in message:
        return False
    return any(k in message for k in _REUPLOAD_RECORD_KEYWORDS)


def _merge_record_basic_info(collected: dict, record_data: dict) -> bool:
    """把病历提取的基础信息合并进 collected，返回是否有变更

    病历只提供基础信息（姓名/性别/年龄/身高/职业/体重），不涉及医疗内容；
    后续用户在对话中提出修改时以用户修改为准（见 _apply_extracted_basic_fields）。
    """
    changed = False
    if record_data.get("patient_name"):
        collected["name"] = record_data["patient_name"]
        changed = True
    if record_data.get("patient_gender"):
        collected["gender"] = record_data["patient_gender"]
        changed = True
    if record_data.get("patient_age"):
        age_raw = record_data["patient_age"]
        try:
            collected["age"] = int(age_raw)
        except (ValueError, TypeError):
            collected["age"] = age_raw
        changed = True
    if record_data.get("patient_height"):
        collected["height"] = record_data["patient_height"]
        changed = True
    if record_data.get("patient_occupation"):
        collected["occupation"] = record_data["patient_occupation"]
        changed = True
    if record_data.get("patient_weight"):
        collected["weight"] = record_data["patient_weight"]
        changed = True
    return changed


def _apply_extracted_basic_fields(collected: dict, result: Any, user_msg: str) -> bool:
    """把对话中结构化提取的基础信息字段应用到 collected，返回是否有变更

    过敏史/既往史仅在消息明确提到相关关键词时写入（防止从无关对话误提取）。
    姓名/性别/年龄/身高/职业/体重等基础信息以用户最新说法为准（用户修改优先）。
    """
    changed = False
    if result.name:
        collected["name"] = result.name
        changed = True
    if result.gender:
        # 性别统一为中文（male/female/m → 男/女），用户在前端看到的是中文
        g = str(result.gender).strip().lower()
        collected["gender"] = {"male": "男", "female": "女", "m": "男", "f": "女"}.get(
            g, result.gender
        )
        changed = True
    if result.age:
        collected["age"] = result.age
        changed = True
    if result.height:
        collected["height"] = result.height
        changed = True
    if result.occupation:
        collected["occupation"] = result.occupation
        changed = True
    if result.weight:
        collected["weight"] = result.weight
        changed = True
    if result.allergy_history not in (None, "") and "过敏" in user_msg:
        collected["allergy_history"] = result.allergy_history
        changed = True
    if result.past_medical_history not in (None, ""):
        has_kw = any(
            k in user_msg for k in ["既往", "病史", "疾病", "手术", "住院", "得过", "大病"]
        )
        if has_kw:
            collected["past_medical_history"] = result.past_medical_history
            changed = True
    if result.chief_complaint:
        collected["chief_complaint"] = result.chief_complaint
        changed = True
    return changed


# 付费前主诉维度（主诉链路是否问清）
CHIEF_COMPLAINT_DIMENSION = "chief_complaint"
# 付费后系统问诊维度（固定顺序）
SYSTEMIC_DIMENSIONS = ["sleep", "diet", "stool", "urine", "emotion", "thermo"]
# 付费后男科针对性追问轮数（系统问诊完成后，按辨证结果查表内症状追问 2-10 轮）
# MIN 为强制保底轮数；达到 MIN 后每轮做「信息充分度」判定，足够即提前转辨证，MAX 兜底强制转
MALE_INQUIRY_MIN_ROUNDS = 2
MALE_INQUIRY_MAX_ROUNDS = 10

# 辨证前的补充信息确认轮：男科追问完成后、正式辨证前，多问一轮「是否有其他补充」，
# 用户回复后（无论有无补充）才进入正式辨证，避免遗漏手术史/用药/家族史等关键信息
SUPPLEMENT_QUESTION = (
    "您的问诊信息已收集完整。在正式辨证前，请确认是否还有其他需要补充的信息？"
    "（如既往手术史、长期用药、家族病史，或其他身体不适）"
    "如有补充，请直接输入补充内容；如没有，请回复\"没有了\"，"
    "我将为您进行正式辨证并开具处方。"
)

# 病历基础信息确认轮：展示提取结果后给患者一个「确认」点选选项
# （确认/修改是封闭选择；修改需带具体信息，让患者直接文字说明，故仅预设"确认"）
MED_RECORD_CONFIRM_OPTIONS = [
    QuestionChoice(
        title="信息核对",
        type="single",
        options=["确认"],
    ),
]


def _systemic_done(progress: dict) -> bool:
    """付费后系统问诊是否全部维度已覆盖"""
    return all(progress.get(d) for d in SYSTEMIC_DIMENSIONS)


def _male_inquiry_done(state: dict[str, Any], progress: dict) -> bool:
    """付费后男科针对性追问 + 补充信息确认轮是否完成（完成才可转辨证）

    无辨证结果（如测试/异常直接进入 UPLOADING_IMAGES）→ 补充确认轮完成即视为完成；
    有辨证结果 → 追问达到 MIN 轮数且（充分度判定通过 或 已达 MAX 兜底），
    且补充信息确认轮完成。
    """
    diagnosis = state.get("preliminary_diagnosis") or state.get("diagnosis") or {}
    if not diagnosis.get("disease"):
        return bool(progress.get("supplement_done"))
    rounds = int(progress.get("male_inquiry", 0))
    if rounds < MALE_INQUIRY_MIN_ROUNDS:
        return False
    male_ok = (
        rounds >= MALE_INQUIRY_MAX_ROUNDS
        or bool(progress.get("male_inquiry_sufficient"))
    )
    return male_ok and bool(progress.get("supplement_done"))


async def _judge_male_inquiry_sufficient(
    orchestrator: LLMOrchestrator,
    diagnosis: dict,
    inquiry_data: dict,
    state: dict[str, Any],
) -> bool:
    """达到最低轮次后，判断男科症状信息是否足以确认辨证/开方

    注入辨证目标 + 查表症状清单 + 已确认症状 + 最近对话；结构化输出判定。
    判定失败容错为"不足"（继续追问），由 MAX 兜底强制转辨证。
    """
    try:
        checklist = tcm_matcher.format_symptom_checklist(
            diagnosis.get("disease", ""), diagnosis.get("syndrome", "")
        )
        symptoms = inquiry_data.get("symptoms") or []
        recent = state.get("messages", [])[-6:]
        recent_text = "\n".join(
            f"{'患者' if m.get('role') == 'user' else '医生'}: {m.get('content', '')}"
            for m in recent
        )
        result = await orchestrator.ainvoke_structured(
            MaleInquirySufficiency,
            [
                {"role": "system", "content": build_male_inquiry_sufficiency_prompt()},
                {"role": "user", "content": (
                    f"辨证目标：{diagnosis.get('disease', '')} / "
                    f"{diagnosis.get('syndrome', '')}\n"
                    f"相关症状清单：\n{checklist or '（无）'}\n\n"
                    f"已确认症状：{symptoms or '暂无'}\n\n"
                    f"最近对话：\n{recent_text or '暂无'}"
                )},
            ],
        )
        if result:
            logger.info(
                "男科追问充分度判定: sufficient=%s missing=%s",
                result.sufficient, result.missing_areas,
            )
            return bool(result.sufficient)
    except Exception as e:
        logger.debug("男科追问充分度判定跳过: %s", e)
    return False


async def _extract_supplement_info(
    orchestrator: LLMOrchestrator, request_message: str, inquiry_data: dict
) -> dict:
    """补充信息确认轮：把用户补充的信息纳入辨证依据

    用 LLM 判断用户是否真的补充了新信息（SupplementExtraction）：
    - has_supplement=true → 新症状进 inquiry.symptoms，回复原文存 inquiry.supplement；
    - has_supplement=false（用户表示"没有了"等）→ 不写入 supplement，避免污染主诉。
    失败容错：提取失败无法判断时保守保留原文，不阻断转辨证。
    """
    msg = (request_message or "").strip()
    if msg:
        try:
            sr = await orchestrator.ainvoke_structured(
                SupplementExtraction,
                [
                    {"role": "system", "content": build_supplement_extraction_prompt()},
                    {"role": "user", "content": msg},
                ],
            )
        except Exception as e:
            logger.debug("补充信息提取跳过: %s", e)
            sr = None
        if sr is None or sr.has_supplement:
            if sr and sr.new_symptoms:
                existing = inquiry_data.get("symptoms") or []
                fresh = [s for s in sr.new_symptoms if s not in existing]
                if fresh:
                    inquiry_data["symptoms"] = existing + fresh
            inquiry_data["supplement"] = msg
    return inquiry_data


async def _run_supplement_round(
    orchestrator: LLMOrchestrator,
    request_message: str,
    inquiry_data: dict,
    progress: dict,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """辨证前的补充信息确认轮统一出口（男科追问完成或系统问诊兜底时调用）

    三种状态：
    1. supplement_done → 已确认过（测试/兼容）→ 直接转辨证；
    2. supplement_pending → 本轮是用户对补充问题的回复 → 提取补充并完成，转辨证；
    3. 均未设 → 发起补充确认轮（返回补充问题 + 选项，不转辨证）。
    返回组装好的 updates。
    """
    if progress.get("supplement_done"):
        updates["inquiry"] = inquiry_data
        updates["inquiry_progress"] = progress
        updates["state"] = SessionState.DIAGNOSIS.value
        updates["response_text"] = (
            "好的，问诊信息与舌面照已收集齐全，下面为您进行正式辨证。"
        )
        updates["response_action"] = ActionType.DIAGNOSIS
        updates["response_data"] = ResponseData()
        return updates

    if progress.get("supplement_pending"):
        inquiry_data = await _extract_supplement_info(
            orchestrator, request_message, inquiry_data
        )
        progress["supplement_done"] = True
        updates["inquiry"] = inquiry_data
        updates["inquiry_progress"] = progress
        updates["state"] = SessionState.DIAGNOSIS.value
        updates["response_text"] = "好的，信息已收集完整，下面为您进行正式辨证。"
        updates["response_action"] = ActionType.DIAGNOSIS
        updates["response_data"] = ResponseData()
        return updates

    progress["supplement_pending"] = True
    updates["inquiry_progress"] = progress
    updates["response_text"] = SUPPLEMENT_QUESTION
    updates["response_action"] = ActionType.CHAT
    # 补充轮不给选项：有补充直接输入、无补充回复"没有了"（见 SUPPLEMENT_QUESTION 话术）
    updates["response_data"] = ResponseData()
    return updates


def _build_male_inquiry_prompt(diagnosis: dict, male_rounds: int) -> str:
    """构建男科针对性追问提示词（按辨证结果查表内症状清单）"""
    disease = diagnosis.get("disease", "")
    syndrome = diagnosis.get("syndrome", "")
    checklist = tcm_matcher.format_symptom_checklist(disease, syndrome) or (
        "-（表内暂无明细，按男科常见表现询问）"
    )
    return (
        "你是一位中医男科专家。系统问诊已完成，下面针对您的男科主诉做进一步确认。\n"
        f"您的辨证结果为：{disease} / {syndrome}\n"
        f"与该辨证相关的症状清单（请逐项向患者确认是否出现及具体表现）：\n{checklist}\n\n"
        "规则：\n"
        "1. 每次只问 1-2 项，聚焦男科症状（性功能、阴囊、尿路、会阴等）；\n"
        "2. 患者已确认/否认的项目不再重复询问；\n"
        "3. 保持「中医分析反馈 → 自然引出下一问」的结构；\n"
        "4. **不要输出'请稍等''正在为您分析''请耐心等待'等让患者等待的话**，"
        "直接提出下一个问题；全部问完也不要总结等待。"
    )


def _build_diagnosis_query(state: dict[str, Any], chief_complaint: str = "") -> str:
    """构建辨证/检索用 query：主诉 + 系统问诊维度文本（追问内容）"""
    query = chief_complaint or state.get("request_message", "")
    inquiry_data = state.get("inquiry") or {}
    dims = "，".join(
        v for k, v in inquiry_data.items() if k in SYSTEMIC_DIMENSIONS and v
    )
    if dims:
        query = f"{query}，{dims}".strip("，")
    return query


def _build_prescription_query(
    state: dict[str, Any], diagnosis: dict, chief_complaint: str = ""
) -> str:
    """构建开方 RAG 检索 query：疾病/证型 + 主诉 + 追问 + 既往病史/过敏史"""
    parts = []
    if diagnosis.get("disease"):
        parts.append(f"疾病：{diagnosis['disease']}")
    if diagnosis.get("syndrome"):
        parts.append(f"证型：{diagnosis['syndrome']}")
    q = chief_complaint or state.get("request_message", "")
    if q:
        parts.append(f"主诉：{q}")
    inquiry_data = state.get("inquiry") or {}
    dims = "，".join(
        v for k, v in inquiry_data.items() if k in SYSTEMIC_DIMENSIONS and v
    )
    if dims:
        parts.append(f"追问：{dims}")
    patient_info = state.get("patient_info") or {}
    past = patient_info.get("past_medical_history")
    allergy = patient_info.get("allergy_history")
    if past:
        parts.append(f"既往病史：{past}")
    if allergy:
        parts.append(f"过敏史：{allergy}")
    return "，".join(parts)


async def perform_diagnosis(
        state: dict[str, Any],
        orchestrator: LLMOrchestrator,
) -> tuple[dict, str, ResponseData]:
    """执行正式辨病辨证（知识库检索 + LLM 结构化输出）

    Returns:
        (diagnosis_dict, response_text, response_data)
    """
    chief_complaint = state.get("chief_complaint") or ""
    request_message = state.get("request_message", "")
    patient_info = state.get("patient_info") or {}
    patient_gender = patient_info.get("gender") if patient_info else None
    patient_age = patient_info.get("age") if patient_info else None

    query = _build_diagnosis_query(state, chief_complaint)

    # 检索知识库（带性别年龄提高命中率）
    knowledge_context = await retrieve_knowledge(
        query=query,
        knowledge_base="both",
        top_k=5,
        gender=patient_gender,
        age=patient_age,
    )

    # 全列表词汇表（LLM 必须从中选病名/证型，不再用关键词候选做硬约束）
    taxonomy_text = tcm_matcher.format_full_taxonomy()

    # 构建辨证提示词（含全列表选择约束）
    diagnosis_prompt = build_diagnosis_prompt(
        patient_info=patient_info,
        chief_complaint=chief_complaint or request_message,
        inquiry_info=state.get("inquiry", {}),
        tongue_analysis=state.get("tongue_analysis") or [],
        face_analysis=state.get("face_analysis") or [],
        knowledge_context=knowledge_context,
        taxonomy_text=taxonomy_text,
    )

    # 追加详细辨病辨证参考（关键词匹配出的最相关疾病/证型明细，辅助 LLM 决策）
    tcm_ctx = tcm_matcher.format_context(query, top_k=5)
    if tcm_ctx:
        diagnosis_prompt = f"{diagnosis_prompt}\n\n{tcm_ctx}"

    # 结构化输出辨证结果（带重试，失败返回 None 走兜底）
    diagnosis = await orchestrator.ainvoke_structured(
        DiagnosisResult,
        [
            {"role": "system", "content": diagnosis_prompt},
            {"role": "user", "content": request_message},
        ],
    )

    if diagnosis is None:
        diagnosis_dict = {
            "disease": "", "syndrome": "待辨证",
            "analysis": "正在分析中...", "treatment_principle": "待定",
        }
    else:
        diagnosis_dict = diagnosis.model_dump()

    # 构建回复（仅展示辨证结果，不含流程性文字）
    disease = diagnosis_dict.get("disease", "")
    syndrome = diagnosis_dict.get("syndrome", "")
    analysis = diagnosis_dict.get("analysis", "")
    principle = diagnosis_dict.get("treatment_principle", "")

    response = f"【辨证结果】\n\n**辨病**：{disease}\n**证型**：{syndrome}\n"
    if principle:
        response += f"**治法**：{principle}\n\n"
    if analysis:
        response += f"**分析**：{analysis}\n\n"
    # 不含流程引导文字，调用方通过 action 字段决定下一步

    return diagnosis_dict, response, ResponseData(
        diagnosis_json={
            "disease": disease,
            "syndrome": syndrome,
        },
        diagnosis_done=True,
    )


# ============================================================
# 条件边路由函数（纯函数，从 state 读路由决策）
# ============================================================

NODE_BY_STATE: dict[str, str] = {
    SessionState.COLLECTING_BASIC.value: "collecting_basic",
    SessionState.INQUIRY.value: "inquiry",
    SessionState.PRELIMINARY_DIAGNOSIS.value: "preliminary_diagnosis",
    SessionState.SELECTING_PATIENT.value: "selecting_patient",
    SessionState.UPLOADING_IMAGES.value: "uploading_images",
    SessionState.DIAGNOSIS.value: "diagnosis",
    SessionState.PRESCRIBING.value: "prescribing",
}

_NODE_NAMES = (
    list(NODE_BY_STATE.values())
    + ["inquiry_greet", "default_chat", "handle_medical_record"]
)


def route_entry(state: dict[str, Any]) -> str:
    """入口路由：病历处理节点优先；DIAGNOSIS+PRESCRIBE 特判；未知状态兜底"""
    # 病历上传/重传：仅当 urls 是「新病历」（与上次处理过的不同）才走病历处理节点。
    # 同批 urls 重发（前端每轮重复带 urls）→ 走正常状态节点，
    # 让 collecting_basic 等正确捕获用户确认/纠正。
    urls = state.get("request_medical_record_urls")
    if urls:
        prev_urls = (state.get("offline_medical_record") or {}).get("_processed_urls")
        if not prev_urls or list(urls) != list(prev_urls):
            return "handle_medical_record"
    current = state.get("state")
    # 病历待确认：上一轮展示了病历信息，本轮处理确认/修改（COLLECTING_BASIC 自行处理）
    if state.get("med_record_pending_confirm") and current != SessionState.COLLECTING_BASIC.value:
        return "handle_medical_record"
    # 用户明确表达重新上传病历（未带图）→ 引导上传
    if _is_reupload_record_intent(state.get("request_message", "")):
        return "handle_medical_record"
    action = state.get("request_action")
    # 对应 _handle_by_state:146-151 特判：DIAGNOSIS 状态 + PRESCRIBE 动作 → 直接开方
    if current == SessionState.DIAGNOSIS.value and action == "PRESCRIBE":
        return "prescribing"
    return NODE_BY_STATE.get(current, "default_chat")


def route_after_collecting(state: dict[str, Any]) -> str:
    """collecting_basic 后：本轮上传了病历 → 只展示不前进；四字段收齐 → 问候"""
    # 病历刚提取轮（skip）不前进，即使字段已收齐（保持现状）
    if state.get("request_medical_record_urls"):
        return END
    if has_enough_patient_info(state.get("patient_info") or {}):
        return "inquiry_greet"
    return END


def route_after_inquiry(state: dict[str, Any]) -> str:
    """inquiry 后：paid=true 即转初步辨证（无需就诊人，无就诊人由该节点追问）；否则 END"""
    if state.get("request_paid"):
        return "preliminary_diagnosis"
    return END


def route_after_uploading(state: dict[str, Any]) -> str:
    """uploading_images 后：舌面照已有 + 系统问诊覆盖 + 男科追问完成 → 辨证；否则 END"""
    has_photos = bool(
        state.get("request_tongue_urls") or state.get("request_face_urls")
        or state.get("tongue_analysis") or state.get("face_analysis")
    )
    progress = state.get("inquiry_progress") or {}
    if has_photos and _systemic_done(progress) and _male_inquiry_done(state, progress):
        # 用户要求重新上传舌面照 → 不转辨证，等重新上传
        if _is_reupload_photo_intent(state.get("request_message", "")):
            return END
        return "diagnosis"
    return END


ENTRY_MAP = {name: name for name in _NODE_NAMES}
COLLECTING_MAP = {"inquiry_greet": "inquiry_greet", END: END}
INQUIRY_MAP = {"preliminary_diagnosis": "preliminary_diagnosis", END: END}
UPLOADING_MAP = {"diagnosis": "diagnosis", END: END}


# ============================================================
# 节点工厂
# ============================================================

def build_nodes(orchestrator: LLMOrchestrator) -> dict[str, Any]:
    """节点工厂：闭包捕获 orchestrator，返回全部节点

    纯构建（不触发 LLM）；节点签名为 `async def node(state) -> dict`（partial update）。
    """

    # ----------------------------------------------------------
    # COLLECTING_BASIC：收集基础信息（性别/年龄/过敏史/既往史）
    # ----------------------------------------------------------
    async def collecting_basic(state: dict[str, Any]) -> dict:
        """收集基础信息：LLM 对话 → 提取字段 → 自动标"无" → 检查四字段

        病历上传/重传由 handle_medical_record 节点统一处理（含首次上传），
        本节点只负责对话收集与放行。
        """
        collected: dict = dict(state.get("patient_info") or {})
        response_data = ResponseData()
        med_pending = state.get("med_record_pending_confirm", False)
        updates: dict[str, Any] = {}

        # 1. LLM 自然对话
        system_prompt = build_system_prompt(SessionState.COLLECTING_BASIC, collected)
        messages = await build_chat_messages(
            system_prompt, state, state.get("request_message", ""), SessionState.COLLECTING_BASIC
        )
        llm_response = await orchestrator.chat(messages)

        # 2. 尝试结构化提取字段（独立 prompt；带助手回复，让 LLM 能判断是否给问答选项）
        changed = False
        try:
            extraction_prompt = build_basic_info_extraction_prompt()
            result = await orchestrator.ainvoke_structured(
                ExtractedPatientInfo,
                [
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": state.get("request_message", "")},
                    {"role": "user", "content": llm_response},
                ],
            )
            if result:
                changed = _apply_extracted_basic_fields(
                    collected, result, state.get("request_message", "")
                )
        except Exception as e:
            logger.debug("基础信息提取跳过: %s", e)
        await _set_choices(
            response_data, orchestrator, state.get("request_message", ""), llm_response
        )

        # 3. LLM 明确询问了但用户未答 → 主动标"无"
        missing_allergy = "allergy_history" not in collected
        missing_past = "past_medical_history" not in collected
        if missing_allergy or missing_past:
            user_msg = state.get("request_message", "")
            llm_asked_allergy = ("过敏" in llm_response
                                 and ("有" in llm_response or "史" in llm_response))
            llm_asked_past = any(
                t in llm_response for t in ["过什么病", "手术", "疾病", "住院", "既往"]
            )
            user_answered_allergy = "过敏" in user_msg
            user_answered_past = any(
                t in user_msg for t in ["既往", "病史", "疾病", "手术", "住院", "得过", "大病"]
            )

            if missing_allergy and llm_asked_allergy and not user_answered_allergy:
                collected["allergy_history"] = "无"
                changed = True
            if missing_past and llm_asked_past and not user_answered_past:
                collected["past_medical_history"] = "无"
                changed = True

        if changed:
            updates["patient_info"] = collected

        # 3.5. 兜底：超过 3 轮仍未收集全的字段自动填"无"
        basic_round = (state.get("collecting_round") or 0) + 1
        updates["collecting_round"] = basic_round
        if basic_round >= 3:
            filled = False
            if collected.get("allergy_history") in (None, ""):
                collected["allergy_history"] = "无"
                filled = True
            if collected.get("past_medical_history") in (None, ""):
                collected["past_medical_history"] = "无"
                filled = True
            if filled:
                updates["patient_info"] = collected

        # 4. 检测 LLM 是否询问线下就诊 → 设置 need_medical_record
        asked_offline = (
            ("线下" in llm_response and "就诊" in llm_response)
            or ("病历" in llm_response and ("处方" in llm_response or "记录" in llm_response))
            or ("去医院" in llm_response and any(t in llm_response for t in ["看", "看过"]))
        )
        if asked_offline:
            response_data.need_medical_record = True

        # 返回已收集的主诉（若有）
        response_data.chief_complaint = collected.get("chief_complaint")

        # 5. 上一轮已设病历确认标记 → 本轮清除
        #    （病历展示轮设置标记；本节点处理 COLLECTING_BASIC 下的确认/修改，非该状态由
        #     handle_medical_record 的确认分支清除）
        if med_pending:
            updates["_deleted_fields"] = ["med_record_pending_confirm"]

        # 四个字段全部收齐才前进（拒绝空字符串和 None）
        if has_enough_patient_info(collected):
            # 进入 INQUIRY：响应文本由 inquiry_greet 节点产出（问候 chat），这里只转状态
            updates["state"] = SessionState.INQUIRY.value
            updates["response_data"] = response_data
            return updates

        updates["response_text"] = llm_response
        updates["response_action"] = ActionType.COLLECT_BASIC_INFO
        updates["response_data"] = response_data
        return updates

    # ----------------------------------------------------------
    # inquiry_greet（新增，必须）：收齐四字段后转 INQUIRY 时的「问候 chat」
    # 对应 consultation_service 中 transition 分支的问候响应，不是完整症状提取
    # ----------------------------------------------------------
    async def inquiry_greet(state: dict[str, Any]) -> dict:
        """四字段收齐后进入问诊阶段的问候对话"""
        collected = state.get("patient_info") or {}
        inquiry_prompt = build_system_prompt(SessionState.INQUIRY, collected)
        transition_messages = await build_chat_messages(
            inquiry_prompt,
            state,
            state.get("request_message", ""),
            SessionState.INQUIRY,
        )
        transition_response = await orchestrator.chat(transition_messages)
        response_data = ResponseData()
        # 问候也常是主诉链路提问（如"晨勃是A、B还是C"）→ 结构化提取问答选项
        await _set_choices(
            response_data, orchestrator, state.get("request_message", ""), transition_response
        )
        return {
            "response_text": transition_response,
            "response_action": ActionType.CHAT,
            "response_data": response_data,
        }

    # ----------------------------------------------------------
    # INQUIRY：问诊阶段（持续收集主诉和症状）
    # ----------------------------------------------------------
    async def inquiry(state: dict[str, Any]) -> dict:
        """问诊阶段（付费前）：主诉链路驱动——识别主诉 → 按链路条件追问 → 引导付费"""
        chief_complaint = state.get("chief_complaint") or ""

        # 检测支付确认：paid=true 即视为已付费（不再要求 hos_sick_info 同时存在；
        # 没确认就诊人时由 PRELIMINARY_DIAGNOSIS/SELECTING_PATIENT 追问 need_select=true）
        request_paid = state.get("request_paid")
        payment_detected = bool(request_paid)

        system_prompt = build_system_prompt(
            SessionState.INQUIRY,
            state.get("patient_info"),
            chief_complaint,
            paid=payment_detected or bool(request_paid),
        )

        # 追加辨病辨证参考（从结构化数据匹配当前症状对应的疾病和证型）
        tcm_ctx = tcm_matcher.format_context(
            chief_complaint or state.get("request_message", ""), top_k=3
        )
        if tcm_ctx:
            system_prompt = f"{system_prompt}\n\n{tcm_ctx}"

        # 追加问诊进度（主诉链路是否已问清）
        progress = dict(state.get("inquiry_progress") or {})
        chain_done = "是" if progress.get(CHIEF_COMPLAINT_DIMENSION) else "否"
        system_prompt += (
            f"\n\n## 问诊进度\n- 主诉链路是否已问清：{chain_done}\n"
            "- 主诉链路核心问题问清后，请在回复中自然引导付费。"
        )

        # LLM 对话收集症状（带知识库上下文和历史）
        messages = await build_chat_messages(
            system_prompt, state, state.get("request_message", ""), SessionState.INQUIRY
        )
        llm_response = await orchestrator.chat(messages)

        # 提取结构化症状信息 + 问诊进度（带容错：LLM 偶尔会胡诌函数名）
        symptom_result = None
        try:
            symptom_result = await orchestrator.ainvoke_structured(
                SymptomExtraction,
                [
                    {"role": "system", "content": build_symptom_extraction_prompt()},
                    {"role": "user", "content": state.get("request_message", "")},
                    {"role": "user", "content": llm_response},
                ],
            )
        except Exception as e:
            logger.warning("症状提取失败（LLM 工具调用异常）: %s", e)
            # 容错：不影响整体对话，LLM 的文本回复仍然可用

        updates: dict[str, Any] = {}

        # 更新主诉：只在为空时写入，保留「用户主动陈述的原始主诉」。
        # 追问确认的 key_findings 不再并入主诉——追问内容归 inquiry 单独保存，
        # 辨证/开方时再分开描述（见 build_diagnosis_prompt 的「主诉/追问采集信息」两段）。
        if symptom_result and symptom_result.key_findings and not chief_complaint:
            chief_complaint = symptom_result.key_findings
            updates["chief_complaint"] = chief_complaint

        # 更新问诊信息
        inquiry_data: dict | None = None
        if symptom_result:
            inquiry_data = {
                "symptoms": symptom_result.symptoms,
                "duration": symptom_result.duration or "",
                "accompanying_symptoms": symptom_result.accompanying_symptoms,
            }
            updates["inquiry"] = inquiry_data
            # 合并问诊进度（主诉链路覆盖）
            if symptom_result.covered_dimensions:
                progress.update({d: True for d in symptom_result.covered_dimensions})
                updates["inquiry_progress"] = progress

        # 检测到支付 → 转入初步辨证（响应交给 preliminary_diagnosis 节点产出）
        if payment_detected:
            # 标记会话为已付费
            updates["paid"] = True
            # 保存就诊人信息（process_chat 已 model_dump 为 dict）
            # 有就诊人才写入（paid=true 不带就诊人时由初步辨证追问；不写 None 避免 Redis 报错）
            if state.get("request_hos_sick_info"):
                updates["hos_sick_info"] = state["request_hos_sick_info"]
            updates["state"] = SessionState.PRELIMINARY_DIAGNOSIS.value
            return updates

        # 未支付：继续问诊，主诉链路问清 → 引导付费
        response_data = ResponseData()
        # 返回本轮已收集的主诉和问诊信息
        response_data.chief_complaint = chief_complaint or None
        latest_inquiry = inquiry_data if symptom_result else state.get("inquiry", {})
        if latest_inquiry:
            response_data.inquiry_json = InquiryJson(**latest_inquiry)
        chain_done_now = bool(symptom_result and symptom_result.chief_complaint_done)
        if chain_done_now or any(kw in llm_response for kw in ["付费", "支付", "费用"]):
            response_data.need_pay = True
        # 问答选项（QuestionChoices 结构化提取）
        await _set_choices(
            response_data, orchestrator, state.get("request_message", ""), llm_response
        )

        updates["response_text"] = llm_response
        updates["response_action"] = ActionType.CHAT
        updates["response_data"] = response_data
        return updates

    # ----------------------------------------------------------
    # PRELIMINARY_DIAGNOSIS：支付后初步辨证（双库检索+初步建议+校验就诊人）
    # ----------------------------------------------------------
    async def preliminary_diagnosis(state: dict[str, Any]) -> dict:
        """初步辨证：支付后双库检索 → 给出初步说明和建议 → 校验就诊人 → 引导上传舌面照"""
        chief_complaint = state.get("chief_complaint") or ""
        request_message = state.get("request_message", "")
        patient_info = state.get("patient_info") or {}
        patient_gender = patient_info.get("gender") if patient_info else None
        patient_age = patient_info.get("age") if patient_info else None

        query = _build_diagnosis_query(state, chief_complaint)

        # 双库检索（带上性别和年龄提高命中率）
        knowledge_context = await retrieve_knowledge(
            query=query,
            knowledge_base="both",
            top_k=5,
            gender=patient_gender,
            age=patient_age,
        )

        # 全列表词汇表（LLM 必须从中选病名/证型）
        taxonomy_text = tcm_matcher.format_full_taxonomy()

        # 构建初步辨证提示词（含全列表选择约束）
        diagnosis_prompt = build_diagnosis_prompt(
            patient_info=patient_info,
            chief_complaint=chief_complaint or request_message,
            inquiry_info=state.get("inquiry", {}),
            knowledge_context=knowledge_context,
            taxonomy_text=taxonomy_text,
        )

        # 追加详细辨病辨证参考
        tcm_ctx = tcm_matcher.format_context(query, top_k=5)
        if tcm_ctx:
            diagnosis_prompt = f"{diagnosis_prompt}\n\n{tcm_ctx}"

        # 结构化输出辨证结果（带重试，失败返回 None 走兜底）
        diagnosis = await orchestrator.ainvoke_structured(
            DiagnosisResult,
            [
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content": request_message},
            ],
        )

        if diagnosis is None:
            diagnosis_dict = {"disease": "", "syndrome": "待辨证", "analysis": "正在为您分析...",
                              "treatment_principle": "待定"}
        else:
            diagnosis_dict = diagnosis.model_dump()

        # 保存初步辨证结果
        updates: dict[str, Any] = {"preliminary_diagnosis": diagnosis_dict}

        # ============================================================
        # 校验就诊人信息（hos_sick_info vs. patient_info）
        # ============================================================
        # 优先取请求中的，否则从会话读取支付时已保存的
        hos_info = state.get("request_hos_sick_info") or state.get("hos_sick_info")

        if hos_info:
            is_match, reason = simple_patient_match(patient_info, hos_info)

            if is_match:
                # 就诊人匹配 → 直接转上传舌面照（跳过 SELECTING_PATIENT）
                updates["patient_info_confirmed"] = hos_info
                updates["state"] = SessionState.UPLOADING_IMAGES.value

                response = (
                    f"根据您提供的信息，初步分析如下：\n\n"
                    f"辨证：{diagnosis_dict.get('syndrome', '待辨证')}\n"
                    f"{diagnosis_dict.get('analysis', '')}\n\n"
                    f"就诊人已确认：{hos_info.get('name', '')}，"
                    f"请拍摄并上传舌照和面照以便进一步精确辨证。"
                )
                updates["response_text"] = response
                updates["response_action"] = ActionType.UPLOAD_IMAGES
                updates["response_data"] = ResponseData(need_upload_image=True)
                return updates

            # 就诊人不匹配 → 转入就诊人选择，返回对比结果
            updates["patient_mismatch"] = True
            updates["patient_select_pending"] = True
            updates["mismatch_reason"] = reason
            updates["state"] = SessionState.SELECTING_PATIENT.value

            response = (
                f"根据您提供的信息，初步分析如下：\n\n"
                f"辨证：{diagnosis_dict.get('syndrome', '待辨证')}\n"
                f"{diagnosis_dict.get('analysis', '')}\n\n"
                f"就诊人信息不匹配：{reason}"
            )
            updates["response_text"] = response
            updates["response_action"] = ActionType.SELECT_PATIENT
            updates["response_data"] = ResponseData(
                patient_mismatch=True,
                mismatch_reason=reason,
            )
            return updates

        # 没有就诊人信息（付费后未选择就诊人）→ 转入就诊人选择，追问 need_select=true
        updates["state"] = SessionState.SELECTING_PATIENT.value

        response = (
            f"根据您提供的信息，初步分析如下：\n\n"
            f"辨证：{diagnosis_dict.get('syndrome', '待辨证')}\n"
            f"{diagnosis_dict.get('analysis', '')}\n\n"
            "请选择本次问诊的就诊人。"
        )
        updates["response_text"] = response
        updates["response_action"] = ActionType.SELECT_PATIENT
        updates["response_data"] = ResponseData(need_select=True)
        return updates

    # ----------------------------------------------------------
    # SELECTING_PATIENT：选择就诊人（校验后端传入的就诊人信息）
    # ----------------------------------------------------------
    async def selecting_patient(state: dict[str, Any]) -> dict:
        """选择就诊人：比对收集信息与选定就诊人

        规则：
          - 无就诊人信息（付费后未选）→ 追问 need_select=true
          - action=SELECT_PATIENT → 确定/切换就诊人，直接比对；不一致 → need_select=true 待确认
          - action=CHAT 且消息修改了收集数据（如"年龄是35岁"）→ 更新收集数据并重比
          - action=CHAT 待确认中 → 语义确认/否认（confirm 放行 / disagree 重新选择）
        """
        # 选定就诊人（request 优先，否则 session）
        if state.get("request_hos_sick_info"):
            confirmed = dict(state["request_hos_sick_info"])
        else:
            session_confirmed = state.get("hos_sick_info")
            if not session_confirmed:
                return {
                    "response_text": "请选择本次问诊的就诊人。",
                    "response_action": ActionType.SELECT_PATIENT,
                    "response_data": ResponseData(need_select=True),
                }
            confirmed = dict(session_confirmed)

        collected = dict(state.get("patient_info") or {})

        # 补全 hos_sick_info 中缺失的过敏史和既往史
        if not confirmed.get("allergy_history") and collected.get("allergy_history"):
            confirmed["allergy_history"] = collected["allergy_history"]
        if not confirmed.get("past_medical_history") and collected.get("past_medical_history"):
            confirmed["past_medical_history"] = collected["past_medical_history"]

        user_msg = state.get("request_message", "")
        action = state.get("request_action", "CHAT")
        pending = state.get("patient_select_pending", False)
        mismatch_reason = state.get("mismatch_reason") or "信息不一致"
        updates: dict[str, Any] = {}

        # ① 用户消息若修改了「收集数据」→ 更新（用户修改为准），后续用新数据重比
        collected_changed = False
        try:
            result = await orchestrator.ainvoke_structured(
                ExtractedPatientInfo,
                [
                    {"role": "system", "content": build_basic_info_extraction_prompt()},
                    {"role": "user", "content": user_msg},
                ],
            )
            if result:
                collected_changed = _apply_extracted_basic_fields(collected, result, user_msg)
        except Exception as e:
            logger.debug("就诊人收集数据修改提取跳过: %s", e)
        if collected_changed:
            updates["patient_info"] = collected

        # ② 分支判定：SELECT_PATIENT（确定/切换）→ 直接比对；
        #    CHAT 修改了收集数据 → 重比；CHAT 待确认中 → 语义确认/否认；否则首次比对
        if action == ActionType.SELECT_PATIENT.value or collected_changed:
            is_match, reason = await orchestrator.match_patient_info(collected, confirmed)
        elif pending:
            semantic, analysis = await orchestrator.confirm_patient_info(
                collected, confirmed, mismatch_reason, user_msg
            )
            if semantic == "disagree":
                # 用户不同意 → 清除 pending，让前端重新选择（need_select=true）
                updates["_deleted_fields"] = ["patient_select_pending"]
                updates["response_text"] = "已取消当前选择，请重新选择就诊人。"
                updates["response_action"] = ActionType.SELECT_PATIENT
                updates["response_data"] = ResponseData(need_select=True)
                return updates
            if semantic == "confirm":
                # 用户确认 → 以选定就诊人为准，更新收集数据后放行
                updates["_deleted_fields"] = ["patient_select_pending"]
                for key in ("gender", "age", "allergy_history", "past_medical_history"):
                    if confirmed.get(key) is not None:
                        collected[key] = confirmed[key]
                updates["patient_info_confirmed"] = confirmed
                updates["patient_info"] = collected
                updates["state"] = SessionState.UPLOADING_IMAGES.value
                updates["response_text"] = (
                    f"就诊人已确认：{confirmed.get('name', '')}，请拍摄并上传舌照和面照。"
                )
                updates["response_action"] = ActionType.UPLOAD_IMAGES
                updates["response_data"] = ResponseData(need_upload_image=True)
                return updates
            # 表达不明确 → 保持状态，简短提示
            updates["response_text"] = (
                f"就诊人：{confirmed.get('name', '')}，{confirmed.get('gender', '')}，"
                f"{confirmed.get('age', '')}岁。请确认是否以上述就诊人信息为准？"
            )
            updates["response_action"] = ActionType.SELECT_PATIENT
            updates["response_data"] = ResponseData(patient_mismatch=True)
            return updates
        else:
            # 首次进入（无待确认）→ 对比信息
            is_match, reason = await orchestrator.match_patient_info(collected, confirmed)

        # ③ 比对结果处理（SELECT_PATIENT / 修改后重比 / 首次进入 共用）
        if is_match:
            # 一致 → 确认放行（不追问）
            updates["patient_info_confirmed"] = confirmed
            updates["state"] = SessionState.UPLOADING_IMAGES.value
            updates["response_text"] = (
                f"就诊人确认无误：{confirmed.get('name', '')}，请拍摄并上传舌照和面照。"
            )
            updates["response_action"] = ActionType.UPLOAD_IMAGES
            updates["response_data"] = ResponseData(need_upload_image=True)
            if pending:
                updates["_deleted_fields"] = ["patient_select_pending"]
            return updates

        # 不一致 → 展示对比，待确认（need_select=true，可切换或修改收集数据）
        collected_name = f"{collected.get('age', '?')}岁" if collected.get("age") else "?"
        response = (
            f"您前面提供的就诊信息为：{collected.get('gender', '?')}，{collected_name}。"
            f"您选择的就诊人信息为：{confirmed.get('gender', '?')}，"
            f"{confirmed.get('age', '?')}岁。"
        )
        updates["patient_mismatch"] = True
        updates["patient_select_pending"] = True
        updates["mismatch_reason"] = reason
        updates["response_text"] = (
            response + "\n您可以通过「选择就诊人」切换，或直接告知我正确的信息以修改。"
        )
        updates["response_action"] = ActionType.SELECT_PATIENT
        updates["response_data"] = ResponseData(
            patient_mismatch=True, mismatch_reason=reason, need_select=True,
        )
        return updates

    # ----------------------------------------------------------
    # UPLOADING_IMAGES：上传舌照/面照
    # ----------------------------------------------------------
    async def uploading_images(state: dict[str, Any]) -> dict:
        """付费后：系统问诊表推进 + 舌面照分析；全部维度覆盖且照片已传 → 转 DIAGNOSIS"""
        tongue_urls = state.get("request_tongue_urls") or []
        face_urls = state.get("request_face_urls") or []
        # 照片"已有"= 本轮新传 或 历史轮已分析持久化（跨轮系统问诊时照片已存，勿重复引导上传）
        has_photos = bool(
            tongue_urls or face_urls
            or state.get("tongue_analysis") or state.get("face_analysis")
        )
        progress = dict(state.get("inquiry_progress") or {})
        updates: dict[str, Any] = {}

        # 1. 本轮带舌面照 → 分析并写入（不立即转诊断）
        if tongue_urls or face_urls:
            tongue_result = await analyze_tongue_images(
                tongue_urls, orchestrator,
                patient_context=state.get("chief_complaint", ""),
            )
            face_result = await analyze_face_images(face_urls, orchestrator)
            updates["image_urls"] = tongue_urls + face_urls
            updates["tongue_analysis"] = tongue_result
            updates["face_analysis"] = face_result
            # 舌面照分析出结论后 → 做一次辨病辨证（收敛），供后续男科针对性追问；
            # 重传舌面照同样走这里 → 重新分析 + 重新辨证
            merged = dict(state)
            merged["tongue_analysis"] = tongue_result
            merged["face_analysis"] = face_result
            converged, _, _ = await perform_diagnosis(merged, orchestrator)
            if converged.get("disease"):
                updates["preliminary_diagnosis"] = converged
                progress["converged_with_inquiry"] = _systemic_done(progress)
                updates["inquiry_progress"] = progress
                logger.info(
                    "舌面照分析后收敛辨证: %s / %s",
                    converged.get("disease"), converged.get("syndrome"),
                )

        # 2. 系统问诊推进（未全部覆盖）
        if not _systemic_done(progress):
            system_prompt = build_system_prompt(
                SessionState.UPLOADING_IMAGES,
                state.get("patient_info"),
                state.get("chief_complaint") or "",
                paid=True,
            )
            covered = [d for d in SYSTEMIC_DIMENSIONS if progress.get(d)]
            pending = [d for d in SYSTEMIC_DIMENSIONS if not progress.get(d)]
            asked_dim = pending[0] if pending else None
            # 舌面照状态：已上传/分析则不再催促上传（除非用户明确要求重新上传）
            photos_ok = bool(
                tongue_urls or face_urls
                or state.get("tongue_analysis") or state.get("face_analysis")
            )
            photos_status = "已上传（请**不要再**催促上传舌面照）" if photos_ok else "未上传"
            system_prompt += (
                "\n\n## 问诊进度\n"
                f"- 舌面照：{photos_status}\n"
                f"- 已覆盖维度：{covered or '无'}\n"
                f"- 待问维度：{pending}\n"
                f"- 本轮**必须**按顺序询问待问维度中的第一个（{asked_dim}），"
                "不要跳问其他维度；用户回答后判断该维度是否已覆盖。"
            )
            messages = await build_chat_messages(
                system_prompt, state, state.get("request_message", ""),
                SessionState.UPLOADING_IMAGES,
            )
            llm_response = await orchestrator.chat(messages)

            # 结构化提取：症状 + 维度覆盖
            symptom_result = None
            try:
                symptom_result = await orchestrator.ainvoke_structured(
                    SymptomExtraction,
                    [
                        {"role": "system", "content": build_symptom_extraction_prompt()},
                        {"role": "user", "content": state.get("request_message", "")},
                        {"role": "user", "content": llm_response},
                    ],
                )
            except Exception as e:
                logger.warning("系统问诊症状提取失败: %s", e)

            # 合并维度文本进 inquiry dict + 进度
            inquiry_data = dict(state.get("inquiry") or {})
            if symptom_result:
                # 只把「本轮新覆盖」的维度文本写入 inquiry，已覆盖维度不覆盖
                # （防止后期轮次误把更早的优质数据覆盖成"未明确提及…"）
                # 且只接受顺序不晚于本轮应问维度的覆盖（防止 LLM 过度报告跳步）
                newly = [
                    d for d in symptom_result.covered_dimensions
                    if d in SYSTEMIC_DIMENSIONS
                    and (asked_dim is None
                         or SYSTEMIC_DIMENSIONS.index(d) <= SYSTEMIC_DIMENSIONS.index(asked_dim))
                    and not progress.get(d)
                ]
                progress.update({d: True for d in newly})
                # 兜底：本轮应问维度用户已回答 → 确定性标记（防 LLM 漏报导致卡住）
                if asked_dim:
                    progress[asked_dim] = True
                if symptom_result.dimension_findings:
                    for dim, txt in symptom_result.dimension_findings.items():
                        if txt and dim in newly:
                            inquiry_data[dim] = txt
                if symptom_result.symptoms:
                    existing = inquiry_data.get("symptoms") or []
                    fresh = [s for s in symptom_result.symptoms if s not in existing]
                    if fresh:
                        inquiry_data["symptoms"] = existing + fresh
                if symptom_result.duration:
                    inquiry_data["duration"] = symptom_result.duration
                if symptom_result.accompanying_symptoms:
                    inquiry_data["accompanying_symptoms"] = symptom_result.accompanying_symptoms
            updates["inquiry"] = inquiry_data
            updates["inquiry_progress"] = progress

            # 全部维度覆盖 + 男科追问完成 + 照片已有 → 转诊断（响应交给 diagnosis 节点产出）
            systemic_just_done = _systemic_done(progress)
            # 系统问诊刚全部覆盖、男科追问未达标、已有辨证结果 → 本回合直接继续男科追问，
            # 不让患者看到"请稍等/接下来为您分析"这类死局（非流式，患者等待无法触发下一步）
            followup_needed = (
                systemic_just_done
                and not _male_inquiry_done(state, progress)
                and bool((state.get("preliminary_diagnosis") or {}).get("disease"))
                and int(progress.get("male_inquiry", 0)) < MALE_INQUIRY_MAX_ROUNDS
                and has_photos
            )
            if not followup_needed:
                if systemic_just_done and _male_inquiry_done(state, progress) and has_photos:
                    updates["state"] = SessionState.DIAGNOSIS.value
                response_data = ResponseData()
                # 未上传 或 用户明确要求重新上传舌面照 → 引导上传（need_upload_image=true）
                if not has_photos or _is_reupload_photo_intent(state.get("request_message", "")):
                    response_data.need_upload_image = True
                await _set_choices(
                response_data, orchestrator, state.get("request_message", ""), llm_response
            )
                updates["response_text"] = llm_response
                updates["response_action"] = ActionType.CHAT
                updates["response_data"] = response_data
                return updates
            # followup_needed → 不返回，直接落入下方男科针对性追问（2.5），
            # 该回合的系统问诊 llm_response 仅用于症状提取，不作为给患者的回复
            logger.info(
                "系统问诊全部覆盖，直接进入男科追问（男科第 %d 轮），不让患者等待",
                int(progress.get("male_inquiry", 0)) + 1,
            )

        # 2.5 系统问诊已覆盖 → 男科针对性追问（按辨证结果查表内症状，2-10 轮）
        diagnosis = state.get("preliminary_diagnosis") or {}
        male_rounds = int(progress.get("male_inquiry", 0))
        if diagnosis.get("disease") and male_rounds < MALE_INQUIRY_MAX_ROUNDS:
            # 若刚从系统问诊落入（2.5 之前系统问诊更新了 inquiry）→ 基于更新后的数据，
            # 否则取历史（系统问诊此前已完成）
            inquiry_data = updates.get("inquiry") or dict(state.get("inquiry") or {})
            # 补充信息确认轮（男科已达标并已发起）→ 提取用户补充并转辨证，不再问男科
            if progress.get("supplement_pending") or progress.get("supplement_done"):
                return await _run_supplement_round(
                    orchestrator, state.get("request_message", ""),
                    inquiry_data, progress, updates,
                )
            # 若收敛辨证是照片上传时做的（早于系统问诊完成）→ 用完整系统问诊数据补收敛
            if _systemic_done(progress) and not progress.get("converged_with_inquiry"):
                merged = dict(state)
                merged["inquiry"] = inquiry_data
                merged["inquiry_progress"] = progress
                re_converged, _, _ = await perform_diagnosis(merged, orchestrator)
                if re_converged.get("disease"):
                    diagnosis = re_converged
                    updates["preliminary_diagnosis"] = re_converged
                progress["converged_with_inquiry"] = True
                updates["inquiry_progress"] = progress
                logger.info("男科追问前补收敛辨证: %s / %s",
                            diagnosis.get("disease"), diagnosis.get("syndrome"))
            male_prompt = _build_male_inquiry_prompt(diagnosis, male_rounds)
            messages = await build_chat_messages(
                male_prompt, state, state.get("request_message", ""),
                SessionState.UPLOADING_IMAGES,
            )
            llm_response = await orchestrator.chat(messages)
            # 收集本轮确认的男科症状进 inquiry dict（供最终辨证）
            symptom_result = None
            try:
                symptom_result = await orchestrator.ainvoke_structured(
                    SymptomExtraction,
                    [
                        {"role": "system", "content": build_symptom_extraction_prompt()},
                        {"role": "user", "content": state.get("request_message", "")},
                        {"role": "user", "content": llm_response},
                    ],
                )
            except Exception as e:
                logger.warning("男科追问症状提取失败: %s", e)
            if symptom_result and symptom_result.symptoms:
                existing = inquiry_data.get("symptoms") or []
                fresh = [s for s in symptom_result.symptoms if s not in existing]
                if fresh:
                    inquiry_data["symptoms"] = existing + fresh
            male_rounds += 1
            progress["male_inquiry"] = male_rounds
            # 达到最低轮次后，判定已收集信息是否足以确认辨证/开方；
            # 足够 → 标记完成并转辨证；不足 → 继续追问（MAX 兜底强制转）
            if (
                male_rounds >= MALE_INQUIRY_MIN_ROUNDS
                and male_rounds < MALE_INQUIRY_MAX_ROUNDS
                and not progress.get("male_inquiry_sufficient")
            ):
                if await _judge_male_inquiry_sufficient(
                    orchestrator, diagnosis, inquiry_data, state
                ):
                    progress["male_inquiry_sufficient"] = True
                    logger.info("男科追问信息充分，提前转辨证（男科第 %d 轮）", male_rounds)
            updates["inquiry"] = inquiry_data
            updates["inquiry_progress"] = progress

            # 男科追问完成判定：达 MIN 且（充分度通过 或 达 MAX 兜底）且照片已有
            male_done = (
                male_rounds >= MALE_INQUIRY_MIN_ROUNDS
                and has_photos
                and (progress.get("male_inquiry_sufficient")
                     or male_rounds >= MALE_INQUIRY_MAX_ROUNDS)
            )
            if male_done:
                # 男科追问完成 → 进入补充信息确认轮（不立即转辨证）
                logger.info("男科追问完成（第 %d 轮），进入补充信息确认轮", male_rounds)
                return await _run_supplement_round(
                    orchestrator, state.get("request_message", ""),
                    inquiry_data, progress, updates,
                )
            # 男科追问未达标 → 继续追问
            response_data = ResponseData()
            if not has_photos or _is_reupload_photo_intent(state.get("request_message", "")):
                response_data.need_upload_image = True
            await _set_choices(
            response_data, orchestrator, state.get("request_message", ""), llm_response
        )
            updates["response_text"] = llm_response
            updates["response_action"] = ActionType.CHAT
            updates["response_data"] = response_data
            return updates

        # 3. 系统问诊已全部覆盖（无男科追问目标 / 男科已达上限兜底）
        # 用户明确要求重新上传舌面照 → 引导重新上传（不转辨证）
        if _is_reupload_photo_intent(state.get("request_message", "")):
            return {
                "response_text": "好的，您可以重新上传舌面照，我来重新为您分析。",
                "response_action": ActionType.CHAT,
                "response_data": ResponseData(need_upload_image=True),
            }
        if not has_photos:
            return {
                "response_text": (
                    "系统问诊已基本完成，请拍摄清晰的舌照和面照，"
                    "确保光线充足、对焦清晰。"
                ),
                "response_action": ActionType.CHAT,
                "response_data": ResponseData(need_upload_image=True),
            }
        # 进入补充信息确认轮（首次发起 / 进行中完成 / 已确认则转辨证）
        inquiry_data = updates.get("inquiry") or dict(state.get("inquiry") or {})
        return await _run_supplement_round(
            orchestrator, state.get("request_message", ""),
            inquiry_data, progress, updates,
        )

    # ----------------------------------------------------------
    # DIAGNOSIS：辨证阶段
    # ----------------------------------------------------------
    async def diagnosis(state: dict[str, Any]) -> dict:
        """辨证阶段：已有辨证结果 → 回答疑问；否则执行正式辨病辨证"""
        patient_info = state.get("patient_info") or {}
        chief_complaint = state.get("chief_complaint") or ""
        request_message = state.get("request_message", "")

        existing_diagnosis = state.get("diagnosis")

        # ═══════════════════════════════════════════════════════════
        # 重新上传舌面照 → 重新分析 + 重新辨证（覆盖旧结果、清旧处方）
        # ═══════════════════════════════════════════════════════════
        reupload_tongue = state.get("request_tongue_urls") or []
        reupload_face = state.get("request_face_urls") or []
        if reupload_tongue or reupload_face:
            logger.info(
                "检测到舌面照重传（DIAGNOSIS）: 舌照 %d 张, 面照 %d 张",
                len(reupload_tongue), len(reupload_face),
            )
            tongue_result = await analyze_tongue_images(
                reupload_tongue, orchestrator, patient_context=chief_complaint
            )
            face_result = await analyze_face_images(reupload_face, orchestrator)
            merged = dict(state)
            merged["tongue_analysis"] = tongue_result
            merged["face_analysis"] = face_result
            diagnosis_dict, response, response_data = await perform_diagnosis(
                merged, orchestrator
            )

            # 重辨证失败（结构化输出空 → 待辨证）：保留旧结果与旧处方，仅更新图片分析
            if (diagnosis_dict.get("syndrome") == "待辨证"
                    and existing_diagnosis and existing_diagnosis.get("disease")):
                logger.warning("舌面照重传重辨证失败，保留原辨证结果")
                return {
                    "tongue_analysis": tongue_result,
                    "face_analysis": face_result,
                    "image_urls": reupload_tongue + reupload_face,
                    "response_text": (
                        "很抱歉，重新上传的舌面照未能完成辨证分析，"
                        "请拍摄更清晰的照片重新上传。您当前的辨证结果保持不变。"
                    ),
                    "response_action": ActionType.DIAGNOSIS,
                    "response_data": ResponseData(
                        diagnosis_json={
                            "disease": existing_diagnosis.get("disease", ""),
                            "syndrome": existing_diagnosis.get("syndrome", ""),
                        },
                        diagnosis_done=True,
                    ),
                }
            return {
                "diagnosis": diagnosis_dict,
                "tongue_analysis": tongue_result,
                "face_analysis": face_result,
                "image_urls": reupload_tongue + reupload_face,
                "prescription": {},  # 辨证结果变化，旧处方失效
                "response_text": response,
                "response_action": ActionType.DIAGNOSIS,
                "response_data": response_data,
            }

        # ═══════════════════════════════════════════════════════════
        # 已有辨证结果 → 正常对话，回答用户关于辨证的疑问
        # ═══════════════════════════════════════════════════════════
        if existing_diagnosis and existing_diagnosis.get("disease"):
            system_prompt = build_system_prompt(
                SessionState.DIAGNOSIS,
                patient_info,
                chief_complaint,
                paid=True,
            )
            system_prompt += build_existing_diagnosis_display_prompt(existing_diagnosis)

            messages = await build_chat_messages(
                system_prompt, state, request_message, SessionState.DIAGNOSIS
            )
            llm_response = await orchestrator.chat(messages)

            response_data = ResponseData(
                diagnosis_json={
                    "disease": existing_diagnosis.get("disease", ""),
                    "syndrome": existing_diagnosis.get("syndrome", ""),
                },
                diagnosis_done=True,
            )
            # 答疑轮若在提问且含备选（如"是继续调理还是复查"）→ 结构化提取问答选项
            await _set_choices(
            response_data, orchestrator, request_message, llm_response
        )
            return {
                "response_text": llm_response,
                "response_action": ActionType.DIAGNOSIS,
                "response_data": response_data,
            }

        # ═══════════════════════════════════════════════════════════
        # 首次进入 → 正式辨证
        # ═══════════════════════════════════════════════════════════
        diagnosis_dict, response, response_data = await perform_diagnosis(state, orchestrator)
        return {
            "diagnosis": diagnosis_dict,
            "response_text": response,
            "response_action": ActionType.DIAGNOSIS,
            "response_data": response_data,
        }

    # ----------------------------------------------------------
    # PRESCRIBING：处方阶段
    # ----------------------------------------------------------
    async def prescribing(state: dict[str, Any]) -> dict:
        """处方阶段：LLM 生成处方（三层策略：推荐主方 / 历史案例 / 知识库参考）"""
        diagnosis = state.get("diagnosis") or {}
        patient_info = state.get("patient_info") or {}
        chief_complaint = state.get("chief_complaint") or ""
        request_message = state.get("request_message", "")

        # 重新上传舌面照 → 重新分析 + 重新辨证（覆盖旧结果），再按新辨证重新开方
        reupload_tongue = state.get("request_tongue_urls") or []
        reupload_face = state.get("request_face_urls") or []
        reuploaded = bool(reupload_tongue or reupload_face)
        if reuploaded:
            logger.info(
                "检测到舌面照重传（PRESCRIBING）: 舌照 %d 张, 面照 %d 张",
                len(reupload_tongue), len(reupload_face),
            )
            tongue_result = await analyze_tongue_images(
                reupload_tongue, orchestrator, patient_context=chief_complaint
            )
            face_result = await analyze_face_images(reupload_face, orchestrator)
            merged = dict(state)
            merged["tongue_analysis"] = tongue_result
            merged["face_analysis"] = face_result
            new_diagnosis, _, _ = await perform_diagnosis(merged, orchestrator)
            # 重辨证失败（结构化输出空 → 待辨证）：保留原辨证结果，仅更新图片分析
            if new_diagnosis.get("syndrome") == "待辨证" and diagnosis.get("disease"):
                logger.warning("舌面照重传重辨证失败，沿用原辨证结果开方")
            else:
                diagnosis = new_diagnosis

        # 检查是否已有处方（防止 DIAGNOSIS 自动调用后重复生成；重传时强制重新开方）
        existing = state.get("prescription")
        reason = None
        if existing and existing.get("drugList") and not reuploaded:
            prescription = existing
        else:
            # 检索 expert 库（性别+年龄±10 过滤）→ 选案并「原封不动」采用处方
            rag_query = _build_prescription_query(state, diagnosis, chief_complaint)
            results = await retrieve_with_filter(
                rag_query,
                gender=(patient_info or {}).get("gender"),
                age=(patient_info or {}).get("age"),
                age_range=10,
                top_k=5,
                fetch_k=30,
                syndromes=[s for s in (diagnosis.get("syndrome") or "").split(",") if s] or None,
            )
            prescription, reason = await prescribe_from_kb(
                diagnosis=diagnosis,
                results=results,
                patient_info=patient_info,
                chief_complaint=chief_complaint or request_message,
                inquiry_info=state.get("inquiry", {}),
                orchestrator=orchestrator,
            )
            logger.info(
                "处方来源: matched=%s query=%s",
                bool(reason and reason.get("matched")), rag_query[:80],
            )

        # 构建回复（有处方 → 药物列表；无匹配 → 提示交医生填写）
        drugs = prescription.get("drugList", [])
        inst = prescription.get("instruction", {})
        need_doctor = False
        if drugs:
            drugs_text = "\n".join(
                f"- {d.get('name', '')} {d.get('number', '')}g" for d in drugs
            )
            response = f"【处方已开具】\n\n{drugs_text}"
            if inst.get("advice"):
                response += f"\n\n医嘱：{inst['advice']}"
            if inst.get("remark"):
                response += f"\n\n备注：{inst['remark']}"
            response += "\n\n请遵医嘱服用，如有不适请及时复诊。"
        else:
            need_doctor = True
            response = (
                f"已为您完成辨证（辨病：{diagnosis.get('disease', '')}，"
                f"证型：{diagnosis.get('syndrome', '')}）。"
                "知识库暂无完全匹配的既往案例，处方将由医生确认后为您填写开具。"
            )

        response_data = ResponseData(
            diagnosis_json={
                "disease": diagnosis.get("disease", ""),
                "syndrome": diagnosis.get("syndrome", ""),
            },
            prescription_json={
                "disease": prescription.get("disease", ""),
                "syndrome": prescription.get("syndrome", ""),
                "drugList": drugs,
                "instruction": inst,
            },
        )
        if need_doctor:
            response_data.need_doctor_prescription = True

        updates = {
            # 图内从 DIAGNOSIS + PRESCRIBE 直接进入时，需把状态推进到 PRESCRIBING
            "state": SessionState.PRESCRIBING.value,
            "prescription": prescription,
            "response_text": response,
            "response_action": ActionType.PRESCRIBE,
            "response_data": response_data,
        }
        if reason is not None:
            # 选案分析原因写入 Redis（白名单持久化，不展示给用户）
            updates["prescription_reason"] = reason
        # 重传舌面照：写回新分析结果与辨证结果
        if reuploaded:
            updates.update({
                "diagnosis": diagnosis,
                "tongue_analysis": tongue_result,
                "face_analysis": face_result,
                "image_urls": reupload_tongue + reupload_face,
            })
        return updates

    # ----------------------------------------------------------
    # handle_medical_record：病历上传 / 重传 / 确认（任意状态下可触发）
    # ----------------------------------------------------------
    async def handle_medical_record(state: dict[str, Any]) -> dict:
        """病历处理节点：多图分析基础信息并展示待确认；或处理确认/修改；或引导重新上传"""
        urls = state.get("request_medical_record_urls") or []
        collected: dict = dict(state.get("patient_info") or {})
        user_msg = state.get("request_message", "")
        updates: dict[str, Any] = {}

        # ── 1. 本轮带病历图片：重新分析并展示，等待用户确认/修改 ──
        if urls:
            logger.info("检测到病历上传/重传（%d 张）: %s", len(urls), urls)
            record_data = await analyze_medical_record(urls, orchestrator)
            # 记录已处理的 urls（route_entry 据此区分"同批重发"与"新病历"）
            record_data["_processed_urls"] = list(urls)
            if _merge_record_basic_info(collected, record_data):
                updates["patient_info"] = collected
            # 标记待确认（下一轮据此处理确认/修改；COLLECTING_BASIC 下由 collecting_basic 处理）
            updates["med_record_pending_confirm"] = True
            updates["offline_medical_record"] = record_data

            info_text = format_medical_record_basic_info(record_data)
            system_prompt = (
                "你是一位专业的中医男科智能体。患者刚刚上传/重新上传了线下病历照片，"
                "以下是提取到的患者基础信息，请向患者展示。\n\n"
                f"## 提取到的信息\n{info_text}\n\n"
                "注意：\n"
                "1. 只展示基础信息（姓名、性别、年龄、身高、职业、体重），"
                "不要复述或询问病历中的具体病情、诊断、处方等内容。\n"
                "2. 身高、体重、职业、姓名等没提取到就不用提及，也不要追问。\n"
                "3. 如果性别或年龄没提取到，需要询问患者补充（这两项为就诊必需信息）。\n"
                "4. 请询问患者以上信息是否有误：有误请直接告知正确的信息，确认无误可回复'确认'。"
            )
            llm_response = await orchestrator.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg or "请核对病历信息"},
            ])
            response_data = ResponseData()
            # 确认/修改是封闭选择 → 直接给「确认」选项（不依赖 LLM 抽取）
            response_data.options = MED_RECORD_CONFIRM_OPTIONS
            updates["response_text"] = llm_response
            updates["response_action"] = ActionType.COLLECT_BASIC_INFO
            updates["response_data"] = response_data
            return updates

        # ── 2. 用户明确表达重新上传病历（未带图）→ 引导上传 ──
        if _is_reupload_record_intent(user_msg):
            updates["response_text"] = (
                "好的，您可以重新上传病历照片，支持一次上传多张。"
                "上传后我会重新为您核对基本信息。"
            )
            updates["response_action"] = ActionType.CHAT
            updates["response_data"] = ResponseData(need_medical_record=True)
            return updates

        # ── 3. 上一轮已展示病历信息 → 本轮处理确认/修改（用户修改为准）──
        if state.get("med_record_pending_confirm"):
            result = None
            try:
                result = await orchestrator.ainvoke_structured(
                    ExtractedPatientInfo,
                    [
                        {"role": "system", "content": build_basic_info_extraction_prompt()},
                        {"role": "user", "content": user_msg},
                    ],
                )
            except Exception as e:
                logger.debug("病历确认/修改提取跳过: %s", e)
            changed = bool(result) and _apply_extracted_basic_fields(collected, result, user_msg)
            if changed:
                updates["patient_info"] = collected
            updates["_deleted_fields"] = ["med_record_pending_confirm"]
            updates["response_text"] = (
                "好的，已按您的修改更新基本信息。" if changed
                else "好的，基本信息确认无误，我们继续。"
            )
            updates["response_action"] = ActionType.CHAT
            updates["response_data"] = ResponseData()
            return updates

        # 兜底（正常不应到达）
        updates["response_text"] = "好的，您可以上传或重新上传病历照片，我来帮您核对基本信息。"
        updates["response_action"] = ActionType.CHAT
        updates["response_data"] = ResponseData(need_medical_record=True)
        return updates

    # ----------------------------------------------------------
    # default_chat：兜底对话
    # ----------------------------------------------------------
    async def default_chat(state: dict[str, Any]) -> dict:
        return {
            "response_text": (
                "您好，我是中医男科智能体，专注阳痿、早泄、男性不育不孕等男科问题。"
                "请问有什么可以帮您？"
            ),
            "response_action": ActionType.CHAT,
            "response_data": ResponseData(),
        }

    return {
        "collecting_basic": collecting_basic,
        "inquiry": inquiry,
        "preliminary_diagnosis": preliminary_diagnosis,
        "selecting_patient": selecting_patient,
        "uploading_images": uploading_images,
        "diagnosis": diagnosis,
        "prescribing": prescribing,
        "inquiry_greet": inquiry_greet,
        "handle_medical_record": handle_medical_record,
        "default_chat": default_chat,
    }


# ============================================================
# 图构建
# ============================================================

def build_consultation_graph(orchestrator: LLMOrchestrator):
    """构建问诊编排状态图

    只做编排，不落盘（无 checkpointer）。跨请求复用 compiled graph 安全（用 ainvoke）。
    """
    nodes = build_nodes(orchestrator)

    graph = StateGraph(ConsultationState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    # 入口路由（含 DIAGNOSIS+PRESCRIBE 特判 + 未知状态兜底）
    graph.add_conditional_edges(START, route_entry, ENTRY_MAP)

    # 自动流转条件边（真正的自动流转只有 3 处）
    graph.add_conditional_edges("collecting_basic", route_after_collecting, COLLECTING_MAP)
    graph.add_conditional_edges("inquiry", route_after_inquiry, INQUIRY_MAP)
    graph.add_conditional_edges("uploading_images", route_after_uploading, UPLOADING_MAP)

    # 其余节点完成时只更新会话状态到下一状态但 END，等用户下一轮再进
    for name in (
        "preliminary_diagnosis",
        "selecting_patient",
        "diagnosis",
        "prescribing",
        "inquiry_greet",
        "handle_medical_record",
        "default_chat",
    ):
        graph.add_edge(name, END)

    return graph.compile()
