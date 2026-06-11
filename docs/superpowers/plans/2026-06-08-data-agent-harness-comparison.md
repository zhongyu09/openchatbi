# Data Agent Harness 对比研究报告 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撰写一份 8000-12000 字的中文深度研究报告，从 Agent Harness 7 维度对比 6 个开源 Data Agent 项目，并为 OpenChatBI 生成改进路线图。

**Architecture:** 单一 Markdown 文件，按 spec 的 6 章结构组织。所有调研数据已在 subagent 返回中就绪，核心工作是整合、评分、分析和建议。每个 Task 对应报告的一个章节。

**Tech Stack:** Markdown, 中文写作

---

## File Map

- Create: `docs/research/data-agent-harness-comparison-2026-06-08.md` — 完整报告

---

### Task 1: 第 1 章 — 研究概述 + 第 2 章 — 项目全景对比

**Files:**
- Create: `docs/research/data-agent-harness-comparison-2026-06-08.md`

- [ ] **Step 1: 创建报告文件，写第 1 章（研究概述）**

包含：
- 标题 + 摘要
- 研究背景与动机（为什么从 harness 视角评估 Data Agent）
- 评估框架说明（7 维度定义 + 0-5 分制标准）
- 研究方法（源码分析 + 文档审阅 + GitHub 数据）
- 局限性声明

- [ ] **Step 2: 写第 2 章（项目全景对比）**

包含：
- 社区指标对比表：6 项目 × (Stars / Forks / Contributors / 最后更新 / Release 节奏 / License)
- 技术栈矩阵：语言 / Agent 框架 / LLM 支持数 / DB 支持数
- 架构定位光谱描述："GUI 工具 → 单轮 LLM → Agent Pipeline → 完整 Harness 平台"

数据来源：
- OpenChatBI: Repomix 分析结果（156 files, LangGraph, 支持 OpenAI/Anthropic, Presto/SQLite/Trino）
- DB-GPT: 18,937 stars, 172 contributors, AWEL, 2026-06-08 最后 push
- Vanna.ai: 23,577 stars, 30 contributors, 自研框架 v2.0, 2026-02-02 最后 push
- Chat2DB: ~25,400 stars, 50 contributors, Spring Boot, 2026-03-22 最后 push
- Dataherald: ~3,635 stars, 18 contributors, LangChain, **2024-07-24 最后 push（已停滞）**
- MindsDB: 39,265 stars, 30+ contributors, Pydantic AI, 2026-05-25 最后 push

- [ ] **Step 3: 验证**

检查：
- 所有 6 个项目都出现在对比表中
- 数据一致性（stars 等数字与调研结果一致）
- 无占位符

- [ ] **Step 4: Commit**

```bash
git add -f docs/research/data-agent-harness-comparison-2026-06-08.md
git commit -m "docs: add chapters 1-2 of data agent harness comparison report"
```

---

### Task 2: 第 3 章 — 维度 1-3 深度分析

**Files:**
- Modify: `docs/research/data-agent-harness-comparison-2026-06-08.md`

- [ ] **Step 1: 写 3.1 Tool Orchestration**

每项目分析要点：
- OpenChatBI: 双层 LangGraph（主图 + Text2SQL 子图），ToolNode 路由，MCP 工具集成
- DB-GPT: AWEL DAG 编排 + ReActAgent + MCP SSE 连接
- Vanna.ai: LLM native tool-calling loop, max 10 iterations, group-based tool access
- Chat2DB: 无 agent loop，单轮 prompt→SQL，MCP Server 暴露自己为工具端
- Dataherald: LangChain ZeroShotAgent + 7 工具，max 15 iterations
- MindsDB: Pydantic AI agent, exploratory→final query plan-execute 模式

横向对比表 + 最佳实践提炼

- [ ] **Step 2: 写 3.2 Error Handling**

每项目分析要点：
- OpenChatBI: SQL 执行 3 次 retry + 错误分类（text2sql_utils.py）
- DB-GPT: AWEL ON ERROR RETRY/FAIL/LOG + JSON recovery + SQL 安全清洗
- Vanna.ai: ErrorRecoveryStrategy 接口（空壳），RETRY/FAIL/SKIP 枚举但默认 FAIL
- Chat2DB: 无自动修正，SSE 流错误清理
- Dataherald: Agent 15 步自纠正，catch_exceptions 统一处理，SQL 提取容错
- MindsDB: MAX_RETRIES=3，null-filled DataFrame 优雅降级，累积错误上下文

- [ ] **Step 3: 写 3.3 Sandbox & Security**

每项目分析要点：
- OpenChatBI: 三级沙箱（RestrictedPython / Local / Docker）
- DB-GPT: 四层隔离模型 + lyric worker 进程隔离 + Docker 后端 + SQL 白名单
- Vanna.ai: 无 SQL 防护，无 Python 沙箱，FileSystem 抽象可注入但默认不安全
- Chat2DB: 无沙箱，CVSS 9.8 已知漏洞（硬编码凭据 + RCE + SQL 注入）
- Dataherald: sqlparse 关键词黑名单过滤 + Fernet 加密连接 URI + Enterprise RBAC
- MindsDB: AST 级查询白名单 + 自动 LIMIT 100 + 表级权限 + 标识符转义

- [ ] **Step 4: 验证**

检查每个维度：
- 6 个项目全覆盖
- 有横向对比表
- 有最佳实践提炼
- 代码/架构证据具体

- [ ] **Step 5: Commit**

```bash
git add -f docs/research/data-agent-harness-comparison-2026-06-08.md
git commit -m "docs: add harness dimensions 1-3 (orchestration, errors, security)"
```

---

### Task 3: 第 3 章 — 维度 4-7 深度分析

**Files:**
- Modify: `docs/research/data-agent-harness-comparison-2026-06-08.md`

- [ ] **Step 1: 写 3.4 Observability**

每项目分析要点：
- OpenChatBI: Python logging，无专门 tracing/metrics
- DB-GPT: 内建 span-based tracing + OpenTelemetry + LLMPerformanceMonitor + CLI 工具
- Vanna.ai: ObservabilityProvider 接口（全面埋点但无默认实现）+ AuditLogger（参数脱敏）
- Chat2DB: Spring Boot 标准日志，无 AI 专项监控
- Dataherald: LangSmith 集成 + intermediate steps 记录 + token 计量
- MindsDB: Langfuse trace + Prometheus metrics + OpenTelemetry collector

- [ ] **Step 2: 写 3.5 Human-in-the-Loop**

每项目分析要点：
- OpenChatBI: LangGraph interrupt 机制 + AskHuman 工具 + search_knowledge
- DB-GPT: 可视化 Canvas 编排 + 对话式交互，但无显式审批门控
- Vanna.ai: 无 confirmation gate，SQL 生成后直接执行
- Chat2DB: 天然 HITL（用户手动执行 SQL），但非设计性安全机制
- Dataherald: Confidence threshold 门控 + golden SQL 飞轮 + Instructions 系统
- MindsDB: Agent 自主执行，MCP 暴露给外部 client 实现 HITL

- [ ] **Step 3: 写 3.6 Memory & State**

每项目分析要点：
- OpenChatBI: langmem 长期记忆 + SQLite checkpoint + ContextManager 摘要压缩
- DB-GPT: 三层认知记忆（Sensory/ShortTerm/LongTerm）+ 重要性评分 + 知识图谱
- Vanna.ai: AgentMemory（tool-usage pattern 向量检索）+ ConversationStore + LlmContextEnhancer
- Chat2DB: LocalCache 进程内 + Milvus/ES schema 索引 + localStorage 前端持久化
- Dataherald: MongoDB + Vector DB + Smart Cache + 无多轮对话
- MindsDB: 会话级 history + Knowledge Base 语义记忆 + 多租户隔离

- [ ] **Step 4: 写 3.7 Eval & Testing**

每项目分析要点：
- OpenChatBI: RunLedger cassette 录放 + 409 个测试函数
- DB-GPT: BenchmarkService（EXACT/CONTAIN/JSON_PATH）+ DB-GPT-Hub 微调评估
- Vanna.ai: 四维评估器（trajectory/output/LLM-as-judge/efficiency）+ ComparisonReport
- Chat2DB: Chat2DB-SQL-7B Spider benchmark 77.3% + 无系统化测试报告
- Dataherald: SimpleEvaluator + EvaluationAgent 双评估器 + 无可复现 benchmark
- MindsDB: Knowledge Base eval（MRR/nDCG/Hit@k）+ 无 NL2SQL 专项 eval

- [ ] **Step 5: 验证**

同 Task 2 Step 4 检查标准

- [ ] **Step 6: Commit**

```bash
git add -f docs/research/data-agent-harness-comparison-2026-06-08.md
git commit -m "docs: add harness dimensions 4-7 (observability, HITL, memory, eval)"
```

---

### Task 4: 第 4 章 — 综合评估矩阵 + 第 5 章 — 改进路线图 + 第 6 章 — 附录

**Files:**
- Modify: `docs/research/data-agent-harness-comparison-2026-06-08.md`

- [ ] **Step 1: 写第 4 章（综合评估矩阵）**

包含：
- 6×7 打分表（基于前面各维度分析的结论打分）
- 总分（7 维度等权求和）
- 各项目一句话 harness 成熟度总结
- 光谱定位总结

评分依据（初步判断，写报告时根据分析调整）：

| 项目 | Orch | Error | Security | Obs | HITL | Memory | Eval | 总分 |
|------|------|-------|----------|-----|------|--------|------|------|
| OpenChatBI | 4 | 3 | 3 | 1 | 4 | 4 | 3 | 22 |
| DB-GPT | 4 | 3 | 4 | 4 | 2 | 5 | 2 | 24 |
| Vanna.ai | 3 | 1 | 1 | 2 | 1 | 4 | 3 | 15 |
| Chat2DB | 0 | 1 | 0 | 1 | 2 | 2 | 1 | 7 |
| Dataherald | 3 | 4 | 3 | 3 | 4 | 2 | 3 | 22 |
| MindsDB | 4 | 3 | 4 | 4 | 2 | 3 | 2 | 22 |

- [ ] **Step 2: 写第 5 章（OpenChatBI 改进路线图）**

基于评分矩阵，OpenChatBI 最低分是 Observability（1 分）：

**P0（可观测性 — 当前最大短板）**
- 借鉴 DB-GPT span-based tracing / MindsDB Langfuse+Prometheus 三件套
- 具体建议：接入 LangSmith 或 Langfuse + 增加 LLM token/cost metrics

**P1（Error Handling + Eval）**
- Error: 借鉴 Dataherald 的错误分类 + 结构化 error code
- Eval: 扩充 RunLedger cases + 借鉴 Vanna 四维评估器

**P2（差异化优势巩固）**
- HITL: 已是标杆（LangGraph interrupt），可参考 Dataherald golden SQL 飞轮增强
- Memory: 已有 langmem，可参考 DB-GPT 三层认知记忆增强

- [ ] **Step 3: 写第 6 章（附录）**

包含：
- 各项目 GitHub 链接 + 数据快照表
- 评估方法论说明
- 术语表（Agent Harness / HITL / AWEL / MCP 等）

- [ ] **Step 4: 全文验证**

检查：
- 报告总字数在 8000-12000 范围
- 所有章节完整，无占位符
- 评分矩阵与各维度分析结论一致
- 改进建议引用了具体的竞品实践

- [ ] **Step 5: Final commit**

```bash
git add -f docs/research/data-agent-harness-comparison-2026-06-08.md
git commit -m "docs: complete data agent harness comparison research report"
```
