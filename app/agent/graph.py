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
    build_diagnosis_prompt,
    build_existing_diagnosis_display_prompt,
    build_symptom_extraction_prompt,
    build_system_prompt,
)
from app.agent.state_machine import STATE_KNOWLEDGE_BASE, ActionType, SessionState
from app.agent.structured_output import (
    DiagnosisResult,
    ExtractedPatientInfo,
    SymptomExtraction,
)
from app.agent.tools import generate_prescription, retrieve_knowledge
from app.knowledge.prescription_index import prescription_index
from app.knowledge.tcm_matcher import tcm_matcher
from app.models.chat_schema import InquiryJson, ResponseData
from app.multimodal.medical_record import analyze_medical_record
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

    query = chief_complaint or request_message

    # 匹配候选证型（tcm_matcher 表内硬约束，防止 LLM 编造表外证型）
    candidate_syndromes = [s["name"] for s in tcm_matcher.match_syndromes(query, top_k=5)]

    # 检索知识库（带上性别年龄提高命中率）
    knowledge_context = await retrieve_knowledge(
        query=query,
        knowledge_base="both",
        top_k=5,
        gender=patient_gender,
        age=patient_age,
    )

    # 构建辨证提示词（含候选证型约束）
    diagnosis_prompt = build_diagnosis_prompt(
        patient_info=patient_info,
        chief_complaint=chief_complaint or request_message,
        inquiry_info=state.get("inquiry", {}),
        tongue_analysis=state.get("tongue_analysis") or [],
        face_analysis=state.get("face_analysis") or [],
        knowledge_context=knowledge_context,
        candidate_syndromes=candidate_syndromes,
    )

    # 追加详细辨病辨证参考（疾病 + 证型明细）
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

_NODE_NAMES = list(NODE_BY_STATE.values()) + ["inquiry_greet", "default_chat"]


def route_entry(state: dict[str, Any]) -> str:
    """入口路由：DIAGNOSIS + PRESCRIBE 特判；未知状态兜底 default_chat"""
    current = state.get("state")
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
    """inquiry 后：检测到支付 → 初步辨证；否则 END"""
    if state.get("request_paid") and state.get("request_hos_sick_info") is not None:
        return "preliminary_diagnosis"
    return END


def route_after_uploading(state: dict[str, Any]) -> str:
    """uploading_images 后：有图 → 辨证；否则 END"""
    if state.get("request_tongue_urls") or state.get("request_face_urls"):
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
        """收集基础信息：检测线下病历上传 → LLM 对话 → 提取字段 → 自动标"无" → 检查四字段"""
        collected: dict = dict(state.get("patient_info") or {})
        response_data = ResponseData()
        med_pending = state.get("med_record_pending_confirm", False)
        skip_post_processing = False  # 病历刚提取时跳过 auto-fill 和 transition
        updates: dict[str, Any] = {}

        # 0. 检测用户是否上传了线下病历照片
        oss_urls = state.get("request_medical_record_urls") or []
        has_uploaded_record = bool(oss_urls)
        medical_context = ""

        if has_uploaded_record:
            logger.info("检测到线下病历上传: %s", oss_urls)
            record_data = await analyze_medical_record(oss_urls[0], orchestrator)
            if record_data.get("summary"):
                logger.info("病历分析摘要: %s", record_data["summary"])

            # 从病历分析中提取所有信息 -> 写入 collected
            changed_med = False
            if record_data.get("patient_gender"):
                collected["gender"] = record_data["patient_gender"]
                changed_med = True
                logger.info("从病历提取性别: %s", record_data["patient_gender"])
            if record_data.get("patient_age"):
                age_raw = record_data["patient_age"]
                try:
                    collected["age"] = int(age_raw)
                except (ValueError, TypeError):
                    collected["age"] = age_raw
                changed_med = True
                logger.info("从病历提取年龄: %s", collected["age"])
            if record_data.get("allergy_history"):
                collected["allergy_history"] = record_data["allergy_history"]
                changed_med = True
                logger.info("从病历提取过敏史: %s", record_data["allergy_history"])
            if record_data.get("past_medical_history"):
                collected["past_medical_history"] = record_data["past_medical_history"]
                changed_med = True
                logger.info("从病历提取既往史: %s", record_data["past_medical_history"])
            if record_data.get("chief_complaint"):
                collected["chief_complaint"] = record_data["chief_complaint"]
                changed_med = True
                logger.info("从病历提取主诉: %s", record_data["chief_complaint"])

            if changed_med:
                updates["patient_info"] = collected

            # 标记需要用户确认（下一轮会读到）
            updates["med_record_pending_confirm"] = True
            updates["offline_medical_record"] = record_data

            # 构建病历完整上下文，供 LLM 向用户展示并请求确认
            ctx_parts = [
                "患者已上传线下病历照片，以下是提取到的信息，请向患者展示并询问是否需要修改或补充。"
            ]
            gender_display = "男" if record_data.get("patient_gender") == "male" else (
                        record_data.get("patient_gender") or "")
            if gender_display:
                ctx_parts.append(f"- 性别: {gender_display}")
            if record_data.get("patient_age"):
                ctx_parts.append(f"- 年龄: {record_data['patient_age']}岁")
            if record_data.get("chief_complaint"):
                ctx_parts.append(f"- 主诉: {record_data['chief_complaint']}")
            if record_data.get("current_symptoms"):
                ctx_parts.append(f"- 当前症状: {record_data['current_symptoms']}")
            if record_data.get("allergy_history"):
                ctx_parts.append(f"- 过敏史: {record_data['allergy_history']}")
            if record_data.get("past_medical_history"):
                ctx_parts.append(f"- 既往史: {record_data['past_medical_history']}")
            if record_data.get("diagnosis"):
                ctx_parts.append(f"- 诊断: {record_data['diagnosis']}")
            if record_data.get("medications"):
                ctx_parts.append(f"- 用药: {record_data['medications']}")
            if record_data.get("summary"):
                ctx_parts.append(f"- 摘要: {record_data['summary']}")
            medical_context = "\n".join(ctx_parts)
            skip_post_processing = True  # 本轮只展示信息请求确认，不做后续处理

        system_prompt = build_system_prompt(SessionState.COLLECTING_BASIC, collected)
        if medical_context:
            system_prompt = f"{system_prompt}\n\n## 线下病历信息\n{medical_context}"

        # 1. LLM 自然对话
        messages = await build_chat_messages(
            system_prompt, state, state.get("request_message", ""), SessionState.COLLECTING_BASIC
        )
        llm_response = await orchestrator.chat(messages)

        # 2. 尝试结构化提取字段（独立 prompt）
        changed = False
        try:
            extraction_prompt = build_basic_info_extraction_prompt()
            result = await orchestrator.ainvoke_structured(
                ExtractedPatientInfo,
                [
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": state.get("request_message", "")},
                ],
            )
            if result:
                if result.gender:
                    collected["gender"] = result.gender
                    changed = True
                if result.age:
                    collected["age"] = result.age
                    changed = True
                if (result.allergy_history not in (None, "")
                        and "过敏" in state.get("request_message", "")):
                    collected["allergy_history"] = result.allergy_history
                    changed = True
                if result.past_medical_history not in (None, ""):
                    has_kw = any(
                        kw in state.get("request_message", "")
                        for kw in ["既往", "病史", "疾病", "手术", "住院"]
                    )
                    if has_kw:
                        collected["past_medical_history"] = result.past_medical_history
                        changed = True
                if result.chief_complaint:
                    collected["chief_complaint"] = result.chief_complaint
                    changed = True
        except Exception as e:
            logger.debug("基础信息提取跳过: %s", e)

        # 3. LLM 明确询问了但用户未答 → 主动标"无"（病历刚提取本轮跳过）
        if not skip_post_processing:
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
                    t in user_msg for t in ["既往", "病史", "疾病", "手术", "住院"]
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
        if asked_offline and not has_uploaded_record:
            response_data.need_medical_record = True

        # 返回已收集的主诉（若有）
        response_data.chief_complaint = collected.get("chief_complaint")

        # 5. 上一轮已设病历确认标记 → 本轮清除（仅当本轮没有新上传时，否则再设 True）
        if med_pending and not has_uploaded_record:
            updates["_deleted_fields"] = ["med_record_pending_confirm"]

        # 6. 病历刚提取一轮（待用户确认）→ 只展示不前进
        if skip_post_processing:
            updates["response_text"] = llm_response
            updates["response_action"] = ActionType.COLLECT_BASIC_INFO
            updates["response_data"] = response_data
            return updates

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
        return {
            "response_text": transition_response,
            "response_action": ActionType.CHAT,
        }

    # ----------------------------------------------------------
    # INQUIRY：问诊阶段（持续收集主诉和症状）
    # ----------------------------------------------------------
    async def inquiry(state: dict[str, Any]) -> dict:
        """问诊阶段：持续收集主诉和症状；检测支付 → 转初步辨证"""
        chief_complaint = state.get("chief_complaint") or ""

        # 检测支付确认：需要 paid=true 且 hos_sick_info 同时存在
        request_paid = state.get("request_paid")
        payment_detected = bool(request_paid) and state.get("request_hos_sick_info") is not None

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

        # LLM 对话收集症状（带知识库上下文和历史）
        messages = await build_chat_messages(
            system_prompt, state, state.get("request_message", ""), SessionState.INQUIRY
        )
        llm_response = await orchestrator.chat(messages)

        # 提取结构化症状信息（带容错：LLM 偶尔会胡诌函数名）
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

        # 更新主诉
        if symptom_result and symptom_result.key_findings:
            if chief_complaint and chief_complaint not in symptom_result.key_findings:
                chief_complaint = chief_complaint + "；" + symptom_result.key_findings
            else:
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

        # 检测到支付 → 转入初步辨证（响应交给 preliminary_diagnosis 节点产出）
        if payment_detected:
            # 标记会话为已付费
            updates["paid"] = True
            # 保存就诊人信息（process_chat 已 model_dump 为 dict）
            updates["hos_sick_info"] = state["request_hos_sick_info"]
            updates["state"] = SessionState.PRELIMINARY_DIAGNOSIS.value
            return updates

        # 未支付：继续问诊，检测是否已引导付费
        response_data = ResponseData()
        # 返回本轮已收集的主诉和问诊信息
        response_data.chief_complaint = chief_complaint or None
        latest_inquiry = inquiry_data if symptom_result else state.get("inquiry", {})
        if latest_inquiry:
            response_data.inquiry_json = InquiryJson(**latest_inquiry)
        if any(kw in llm_response for kw in ["付费", "支付", "费用"]):
            response_data.need_pay = True

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

        query = chief_complaint or request_message

        # 匹配候选证型（tcm_matcher 表内硬约束，防止 LLM 编造表外证型）
        candidate_syndromes = [s["name"] for s in tcm_matcher.match_syndromes(query, top_k=5)]

        # 双库检索（带上性别和年龄提高命中率）
        knowledge_context = await retrieve_knowledge(
            query=query,
            knowledge_base="both",
            top_k=5,
            gender=patient_gender,
            age=patient_age,
        )

        # 构建初步辨证提示词（含候选证型约束）
        diagnosis_prompt = build_diagnosis_prompt(
            patient_info=patient_info,
            chief_complaint=chief_complaint or request_message,
            inquiry_info=state.get("inquiry", {}),
            knowledge_context=knowledge_context,
            candidate_syndromes=candidate_syndromes,
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

        # 没有就诊人信息（不应发生）→ 转入就诊人选择
        updates["state"] = SessionState.SELECTING_PATIENT.value

        response = (
            f"根据您提供的信息，初步分析如下：\n\n"
            f"辨证：{diagnosis_dict.get('syndrome', '待辨证')}\n"
            f"{diagnosis_dict.get('analysis', '')}"
        )
        updates["response_text"] = response
        updates["response_action"] = ActionType.SELECT_PATIENT
        updates["response_data"] = ResponseData()
        return updates

    # ----------------------------------------------------------
    # SELECTING_PATIENT：选择就诊人（校验后端传入的就诊人信息）
    # ----------------------------------------------------------
    async def selecting_patient(state: dict[str, Any]) -> dict:
        """选择就诊人：对比 AI 收集的基础信息与后端选择的就诊人，请求用户确认"""
        # 优先取请求中的，否则从 session 读取支付时已保存的
        if state.get("request_hos_sick_info"):
            confirmed = dict(state["request_hos_sick_info"])
        else:
            session_confirmed = state.get("hos_sick_info")
            if not session_confirmed:
                return {
                    "response_text": "请选择就诊人信息。",
                    "response_action": ActionType.SELECT_PATIENT,
                    "response_data": ResponseData(),
                }
            confirmed = dict(session_confirmed)

        collected = dict(state.get("patient_info") or {})

        # 补全 hos_sick_info 中缺失的过敏史和既往史
        if not confirmed.get("allergy_history") and collected.get("allergy_history"):
            confirmed["allergy_history"] = collected["allergy_history"]
        if not confirmed.get("past_medical_history") and collected.get("past_medical_history"):
            confirmed["past_medical_history"] = collected["past_medical_history"]

        # 检查是否已有待确认的比对
        pending = state.get("patient_select_pending", False)

        if pending:
            # 用户已看到比对结果，本轮是用户的确认/否认回复
            user_msg = state.get("request_message", "")
            # 语义分析：用 LLM 理解用户真实意图（代替关键词匹配）
            mismatch_reason = state.get("mismatch_reason") or "信息不一致"
            action, analysis = await orchestrator.confirm_patient_info(
                collected, confirmed, mismatch_reason, user_msg
            )

            if action == "disagree":
                # 用户不同意 → 清除 pending，让前端重新选择
                return {
                    "_deleted_fields": ["patient_select_pending"],
                    "response_text": "已取消当前选择，请重新选择就诊人。",
                    "response_action": ActionType.SELECT_PATIENT,
                    "response_data": ResponseData(need_select=True),
                }

            if action == "confirm":
                # 用户已确认 → 以 hos_sick_info 为准，更新 patient_info
                updated_info = dict(collected)
                for key in ("gender", "age", "allergy_history", "past_medical_history"):
                    if confirmed.get(key) is not None:
                        updated_info[key] = confirmed[key]
                response = f"就诊人已确认：{confirmed.get('name', '')}，请拍摄并上传舌照和面照。"
                return {
                    "_deleted_fields": ["patient_select_pending"],
                    "patient_info_confirmed": confirmed,
                    "patient_info": updated_info,
                    "state": SessionState.UPLOADING_IMAGES.value,
                    "response_text": response,
                    "response_action": ActionType.UPLOAD_IMAGES,
                    "response_data": ResponseData(need_upload_image=True),
                }

            # 表达不明确 → 保持状态，简短提示
            return {
                "response_text": (
                    f"就诊人：{confirmed.get('name', '')}，{confirmed.get('gender', '')}，"
                    f"{confirmed.get('age', '')}岁。请确认是否以上述就诊人信息为准？"
                ),
                "response_action": ActionType.SELECT_PATIENT,
                "response_data": ResponseData(patient_mismatch=True),
            }

        # --- 首次进入：对比信息 ---
        is_match, reason = await orchestrator.match_patient_info(collected, confirmed)

        if not is_match:
            # 有不一致 → 展示对比，请求确认
            collected_name = f"{collected.get('age', '?')}岁" if collected.get("age") else "?"
            collected_gender = collected.get("gender", "?")
            confirmed_gender = confirmed.get("gender", "?")
            confirmed_age = confirmed.get("age", "?")

            response = (
                f"您前面提供的就诊信息为：{collected_gender}，{collected_name}。"
                f"您选择的就诊人信息为：{confirmed_gender}，{confirmed_age}岁。"
            )
            return {
                "patient_mismatch": True,
                "patient_select_pending": True,
                "mismatch_reason": reason,
                "response_text": response,
                "response_action": ActionType.SELECT_PATIENT,
                "response_data": ResponseData(
                    patient_mismatch=True, mismatch_reason=reason,
                ),
            }

        # 完全一致 → 直接放行
        response = f"就诊人确认无误：{confirmed.get('name', '')}，请拍摄并上传舌照和面照。"
        return {
            "patient_info_confirmed": confirmed,
            "state": SessionState.UPLOADING_IMAGES.value,
            "response_text": response,
            "response_action": ActionType.UPLOAD_IMAGES,
            "response_data": ResponseData(need_upload_image=True),
        }

    # ----------------------------------------------------------
    # UPLOADING_IMAGES：上传舌照/面照
    # ----------------------------------------------------------
    async def uploading_images(state: dict[str, Any]) -> dict:
        """上传舌照/面照阶段；有图则分析并转 DIAGNOSIS（辨证由 diagnosis 节点执行）"""
        tongue_urls = state.get("request_tongue_urls") or []
        face_urls = state.get("request_face_urls") or []

        if tongue_urls or face_urls:
            # 分别单独分析舌照和面照
            tongue_result = await analyze_tongue_images(
                tongue_urls, orchestrator,
                patient_context=state.get("chief_complaint", ""),
            )
            face_result = await analyze_face_images(face_urls, orchestrator)

            # 分析结果写入 state，diagnosis 节点直接读取（快照 trick 自然消失）
            return {
                "image_urls": tongue_urls + face_urls,
                "tongue_analysis": tongue_result,
                "face_analysis": face_result,
                "state": SessionState.DIAGNOSIS.value,
            }

        # 无图片，引导上传
        return {
            "response_text": "请拍摄清晰的舌照和面照，确保光线充足、对焦清晰。",
            "response_action": ActionType.CHAT,
            "response_data": ResponseData(need_upload_image=True),
        }

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

            return {
                "response_text": llm_response,
                "response_action": ActionType.DIAGNOSIS,
                "response_data": ResponseData(
                    diagnosis_json={
                        "disease": existing_diagnosis.get("disease", ""),
                        "syndrome": existing_diagnosis.get("syndrome", ""),
                    },
                    diagnosis_done=True,
                ),
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
        patient_gender = patient_info.get("gender") if patient_info else None
        patient_age = patient_info.get("age") if patient_info else None

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
        if existing and existing.get("drugList") and not reuploaded:
            prescription = existing
        else:
            # 处方阶段也检索知识库（带上性别年龄提高命中率）
            kb_type = STATE_KNOWLEDGE_BASE.get(SessionState.PRESCRIBING, "general")
            knowledge_context = await retrieve_knowledge(
                query=chief_complaint or request_message,
                knowledge_base=kb_type,
                top_k=3,
                gender=patient_gender,
                age=patient_age,
            )

            # 获取推荐主方（从 tcm_syndrome.recommended_formula）
            syndrome_name = diagnosis.get("syndrome", "")
            disease_name = diagnosis.get("disease", "")
            base_formula = tcm_matcher.get_recommended_formula(syndrome_name)

            # 证型为空或查不到推荐主方 → 降级：用主诉/症状重新匹配候选证型（防断链）
            if not base_formula:
                fallback_syndromes = tcm_matcher.match_syndromes(
                    chief_complaint or request_message, top_k=1
                )
                if fallback_syndromes:
                    fallback = fallback_syndromes[0]["name"]
                    logger.info(
                        "证型 %r 无推荐主方，降级匹配到候选证型 %s", syndrome_name, fallback
                    )
                    base_formula = tcm_matcher.get_recommended_formula(fallback)
                    if not syndrome_name:
                        # 辨证结果缺失证型时，用降级证型回填（影响后续 prompt 与响应）
                        diagnosis = {**diagnosis, "syndrome": fallback}
                        syndrome_name = fallback

            # 获取相似历史处方案例（从 PrescriptionIndex，按性别年龄过滤）
            similar_prescriptions: list[dict] = []
            if prescription_index.loaded:
                matched = prescription_index.lookup(
                    disease_name, syndrome_name, top_k=3,
                    gender=patient_gender, age=patient_age,
                )
                similar_prescriptions = [
                    {
                        "herbs": m.get("herbs", []),
                        "dosage": m.get("dosage", ""),
                        "age": m.get("age"),
                        "gender": m.get("gender", ""),
                    }
                    for m in matched
                ]

            # 调试日志
            logger.info(
                "处方生成参考: disease=%s syndrome=%s base_formula=%s similar_cases=%d",
                disease_name, syndrome_name,
                base_formula or "(无)",
                len(similar_prescriptions),
            )

            # 生成处方
            prescription = await generate_prescription(
                diagnosis=diagnosis,
                chief_complaint=chief_complaint or request_message,
                patient_info=patient_info,
                knowledge_context=knowledge_context,
                base_formula=base_formula,
                similar_prescriptions=similar_prescriptions,
                orchestrator=orchestrator,
            )

        # 构建回复
        drugs_text = "\n".join(
            f"- {d.get('name', '')} {d.get('number', '')}g"
            for d in prescription.get("drugList", [])
        )
        inst = prescription.get("instruction", {})
        response = f"【处方已开具】\n\n{drugs_text}"
        if inst.get("advice"):
            response += f"\n\n医嘱：{inst['advice']}"
        if inst.get("remark"):
            response += f"\n\n备注：{inst['remark']}"
        response += "\n\n请遵医嘱服用，如有不适请及时复诊。"

        updates = {
            # 图内从 DIAGNOSIS + PRESCRIBE 直接进入时，需把状态推进到 PRESCRIBING
            "state": SessionState.PRESCRIBING.value,
            "prescription": prescription,
            "response_text": response,
            "response_action": ActionType.PRESCRIBE,
            "response_data": ResponseData(
                diagnosis_json={
                    "disease": diagnosis.get("disease", ""),
                    "syndrome": diagnosis.get("syndrome", ""),
                },
                prescription_json={
                    "disease": prescription.get("disease", ""),
                    "syndrome": prescription.get("syndrome", ""),
                    "drugList": prescription.get("drugList", []),
                    "instruction": inst,
                },
            ),
        }
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
    # default_chat：兜底对话
    # ----------------------------------------------------------
    async def default_chat(state: dict[str, Any]) -> dict:
        return {
            "response_text": "您好，我是中医AI助手。请告诉我您的症状。",
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
        "default_chat",
    ):
        graph.add_edge(name, END)

    return graph.compile()
