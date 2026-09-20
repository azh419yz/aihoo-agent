# AI 中医助手 Agent

AI 中医助手 Agent：FastAPI 后端，被 Java 后端通过 HTTP 调用，完成「收集信息 → 问诊 → 辨病辨证 → 开方」的强编排流程。

## 技术栈

- **模型**：统一 `qwen3.8-max`（原生多模态，文本/图片/结构化输出共用），openai SDK 调百炼 compatible-mode 企业网关，全链路 `enable_thinking: false`
- **编排**：LangGraph StateGraph 状态机（10 节点 + 4 条件边），无 LangChain 消息层
- **持久化**：Redis 主存储 + MySQL 兜底恢复（纯 Pydantic + 手写 SQL，无 ORM）
- **知识库**：`tcm_disease` / `tcm_syndrome` 表（辨病辨证标准词汇，LLM 从中选病名/证型）+ 百炼知识库 expert 历史案例（开方选案、处方原样采用）

## 快速开始

```bash
# 1. 安装依赖（uv 管理）
uv sync

# 2. 配置环境变量
cp .env.example .env   # 填入 DASHSCOPE_API_KEY / MySQL / Redis

# 3. 初始化数据库（表结构 + 初始数据；tcm_prescription 表已退役，无需导入）
mysql -u root -p < scripts/init_db.sql
mysql -u root -p aihoo_agent < scripts/tcm_disease.sql
mysql -u root -p aihoo_agent < scripts/tcm_syndrome.sql

# 4. 启动服务
#    本地调试：
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#    生产/重启：bash scripts/restart.sh（停旧进程 → nohup 启动 → 健康检查）
```

服务启动后：API `http://localhost:8000`，Swagger `/docs`，健康检查 `/health`。

## 核心 API

统一返回 `{code, message, data, request_id}` 结构，`code=0` 表示成功。

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/agent/session` | POST | 创建会话（`session_id` / `patient_id`） |
| `/api/v1/agent/chat` | POST | 问诊对话（主流程：收集信息 → 问诊 → 辨证 → 开方） |
| `/api/v1/agent/session/{id}` | GET | 查询会话状态与信息 |
| `/api/v1/agent/session/{id}/messages` | GET | 查询对话消息记录 |
| `/health` | GET | 健康检查 |

`chat` 关键请求字段：`message`、`action`（`CHAT` / `PRESCRIBE`）、`paid`、`hos_sick_info`（就诊人信息）、`tongue_urls` / `face_urls`（舌面照/面照）、`medical_record_urls`（病历图片）。完整字段见 Swagger `/docs`。

## 问诊流程（状态机）

LangGraph StateGraph 驱动，10 个节点 + 4 条条件边。状态流转：

```
未付费：COLLECTING_BASIC（基础四字段）→ INQUIRY（症状问诊）
付费后：PRELIMINARY_DIAGNOSIS（初步辨证）→ SELECTING_PATIENT → UPLOADING_IMAGES（舌面照/面照）→ DIAGNOSIS（正式辨证）→ PRESCRIBING（开方）
```

- `SELECTING_PATIENT` 仅在就诊人与已收集信息不匹配时进入；匹配则直接跳过。
- 辨病辨证基于 `tcm_disease` / `tcm_syndrome` 标准词汇（LLM 从中选病名/证型）；开方从 expert 知识库检索历史案例（性别 + 年龄±10 过滤），选中后**处方原样采用**；知识库无匹配案例时返回空处方并交医生填写（`need_doctor_prescription`）。
- 非法状态流转返回 `INVALID_STATE_TRANSITION`（2002）。

## 测试

```bash
uv run pytest          # 全部测试（MockOrchestrator，不打网络、不连 Redis/MySQL）
uv run ruff check .    # lint（E/F/I/W，line-length 100）
```

## 目录结构

```
app/
├── agent/      图编排（graph / orchestrator / prompts / tools / structured_output / state_machine）
├── api/        路由（薄 controller）
├── common/     共享基础（统一响应 / 响应码 / 异常处理器 / 依赖注入）
├── core/       配置（config.py）
├── knowledge/  TCM 知识库（辨病辨证匹配 / 百炼检索）
├── models/     数据模型（domain 持久化模型 + chat_schema API 契约）
├── multimodal/ 图片分析（舌面照 / 病历）
├── service/    业务服务（问诊编排 / 会话）
└── storage/    存储层（MySQL / Redis）

scripts/        数据库初始化（init_db.sql 全量 DDL + tcm_disease/syndrome 两个初始数据）与 restart.sh 重启脚本
```
