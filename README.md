# AI 中医助手 Agent

AI 中医助手 Agent：FastAPI 后端，被 Java 后端通过 HTTP 调用，完成「收集信息 → 问诊 → 辨病辨证 → 开方」的强编排流程。

## 技术栈

- **模型**：统一 `qwen3.8-max`（原生多模态，文本/图片/结构化输出共用），openai SDK 调百炼 compatible-mode 企业网关，全链路 `enable_thinking: false`
- **编排**：LangGraph StateGraph 状态机（9 节点 + 4 条件边），无 LangChain 消息层
- **持久化**：Redis 主存储 + MySQL 兜底恢复（纯 Pydantic + 手写 SQL，无 ORM）
- **知识库**：`tcm_disease` / `tcm_syndrome` / `tcm_prescription` 表（辨病辨证硬约束）+ 百炼知识库检索

## 快速开始

```bash
# 1. 安装依赖（uv 管理）
uv sync

# 2. 配置环境变量
cp .env.example .env   # 填入 DASHSCOPE_API_KEY / MySQL / Redis

# 3. 初始化数据库（表结构 + 初始数据）
mysql -u root -p < scripts/init_db.sql
mysql -u root -p aihoo_agent < scripts/tcm_disease.sql
mysql -u root -p aihoo_agent < scripts/tcm_syndrome.sql
mysql -u root -p aihoo_agent < scripts/tcm_prescription.sql

# 4. 启动服务
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后：API `http://localhost:8000`，Swagger `/docs`，健康检查 `/health`。

## 测试

```bash
uv run pytest          # 全部测试（MockOrchestrator，不打网络、不连 Redis/MySQL）
```

## 目录结构

```
app/
├── agent/      图编排（graph / orchestrator / prompts / tools / structured_output / state_machine）
├── api/        路由（薄 controller）
├── common/     共享基础（统一响应 / 异常 / 依赖注入）
├── core/       配置（config.py）
├── knowledge/  TCM 知识库（辨病辨证匹配 / 处方索引 / 百炼检索）
├── models/     数据模型（domain 持久化模型 + chat_schema API 契约）
├── multimodal/ 图片分析（舌面照 / 病历）
├── service/    业务服务（问诊编排 / 会话）
└── storage/    存储层（MySQL / Redis）

scripts/        数据库初始化（init_db.sql 全量 DDL + 三个初始数据文件）
```
