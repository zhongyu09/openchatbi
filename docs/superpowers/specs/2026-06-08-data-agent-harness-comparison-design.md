# 开源 Data Agent Harness 架构对比研究 — 设计文档

## 研究目标

以《Anatomy of an AI Harness》7 维度评估框架为标尺，对比分析社区热度和成熟度较高的开源 Data Agent 项目，生成一份专业深度研究报告。最终落地到 OpenChatBI 的改进路线图。

## 评估框架

沿用《Anatomy of an AI Harness》的 7 个核心维度：

| # | 维度 | 评估要点 |
|---|------|----------|
| 1 | Tool Orchestration | 工具路由、多步推理、plan-execute 模式、tool access control |
| 2 | Error Handling | 错误分类、重试机制、自愈能力、优雅降级 |
| 3 | Sandbox & Security | 代码/SQL 执行隔离、注入防护、权限控制、审计 |
| 4 | Observability | 分布式追踪、LLM metrics、成本监控、日志 |
| 5 | Human-in-the-Loop | 审批门控、clarification flow、反馈闭环 |
| 6 | Memory & State | 对话持久化、长期记忆、上下文管理、多租户 |
| 7 | Eval & Testing | 基准测试、回归评估、覆盖率、eval 自动化 |

每个维度采用 0-5 分制量化评分：
- 0 = 无实现
- 1 = 最低限度 / 空壳接口
- 2 = 基础实现，有明显缺陷
- 3 = 功能完整，可用于非关键场景
- 4 = 生产级，覆盖主要边界情况
- 5 = 业界标杆，可作为最佳实践参考

## 对比项目

选择标准：GitHub Stars > 3k，有明确的 Data Agent / NL-to-SQL 能力，社区活跃或架构有参考价值。

| 项目 | Stars | 技术栈 | 定位 | 纳入理由 |
|------|-------|--------|------|----------|
| **OpenChatBI** | — | Python, LangGraph | LangGraph Text2SQL + 数据分析 Agent | 主角，评估基准线 |
| **DB-GPT** | 18.9k | Python, AWEL | 多 Agent 数据交互平台 | 功能最完整的同类竞品，AWEL 编排 + 四层沙箱 |
| **Vanna.ai** | 23.6k | Python, 自研框架 | RAG→Agent Text2SQL 框架 | v2.0 架构转型典型案例，记忆系统创新 |
| **Chat2DB** | 25.4k | Java/TS, Spring Boot | AI 数据库 GUI 客户端 | 高社区热度的非 Agent 参照物 |
| **Dataherald** | 3.6k | Python, LangChain | 企业 NL-to-SQL Agent | Schema linking 最精良，golden SQL 飞轮 |
| **MindsDB** | 39.3k | Python, Pydantic AI | AI-in-Database 平台 | 安全和可观测性标杆，最高 star 数 |

## 报告结构

### 第 1 章：研究概述
- 研究背景：为什么从 harness 视角评估 Data Agent
- 评估框架说明：Anatomy of an AI Harness 7 维度的定义和评分标准
- 研究方法：源码分析 + 文档审阅 + GitHub 数据（非实际运行测试）
- 局限性声明：基于公开源码和文档的静态分析，未做运行时性能测试

### 第 2 章：项目全景对比
内容：
- **社区指标对比表**：Stars、Forks、Contributors、活跃度（最后 commit / release 节奏）、License
- **技术栈矩阵**：编程语言、Agent 框架（LangGraph / AWEL / Pydantic AI / 自研 / 无）、LLM 支持数、数据库支持数
- **架构定位图**（文字描述）：从"LLM 辅助工具"到"自主 Agent 平台"的光谱定位

### 第 3 章：Harness 七维度深度分析（核心章节）
每个维度统一结构：
1. **维度定义** — 一句话说清评什么，为什么重要
2. **横向对比表** — 6 个项目在该维度的关键特征一览
3. **逐项目深度分析** — 具体实现细节、代码/架构证据、设计亮点与缺陷
4. **最佳实践提炼** — 该维度的标杆做法和关键设计原则

#### 3.1 Tool Orchestration
重点对比：
- 是否有 agent loop（plan-execute-reflect）
- 工具路由机制（LLM function calling vs 规则路由 vs 无）
- 多步推理深度（最大迭代次数、中间状态管理）
- 工具权限控制（can_use_tool / group-based access）
- 关键差异：Chat2DB 无 agent loop vs DB-GPT AWEL DAG vs Vanna tool-calling loop

#### 3.2 Error Handling
重点对比：
- SQL 执行失败的重试策略（自动 vs 依赖 LLM 自发 vs 无）
- 错误分类粒度（syntax/permission/timeout vs 粗粒度 vs 无分类）
- LLM 调用失败的 recovery（retry / fallback / circuit breaker）
- 关键差异：Dataherald 15 步 agent 自纠正 vs OpenChatBI 3 次 SQL retry vs Vanna 空壳 recovery

#### 3.3 Sandbox & Security
重点对比：
- SQL 注入防护（AST 级 vs 关键词黑名单 vs 无）
- 代码执行隔离（进程级 / Docker / RestrictedPython / 无）
- 权限控制（表级 / 操作级 / 用户级）
- 已知安全漏洞
- 关键差异：DB-GPT 四层沙箱 vs MindsDB AST 白名单 vs Chat2DB CVSS 9.8 漏洞

#### 3.4 Observability
重点对比：
- 分布式追踪（OpenTelemetry / LangSmith / Langfuse / 无）
- LLM 指标（token 计量、TTFT、成本监控）
- 审计日志（参数脱敏、合规性）
- 关键差异：DB-GPT 内建 span tracing vs MindsDB Langfuse+Prometheus+OTel vs Vanna 有接口无电池

#### 3.5 Human-in-the-Loop
重点对比：
- 审批门控（执行前确认 / 危险操作拦截）
- Clarification flow（主动提问澄清需求）
- 反馈闭环（golden SQL / confidence threshold / 用户纠正）
- 关键差异：OpenChatBI LangGraph interrupt vs Dataherald confidence 门控 + golden SQL vs 多数项目缺失

#### 3.6 Memory & State
重点对比：
- 对话持久化（SQLite / MongoDB / 浏览器 / 无）
- 长期记忆（向量检索 / 知识图谱 / langmem）
- 上下文管理（摘要压缩 / 滑动窗口 / token 预算）
- 多租户隔离
- 关键差异：DB-GPT 三层认知记忆 vs Vanna tool-usage pattern 记忆 vs OpenChatBI langmem 长期记忆

#### 3.7 Eval & Testing
重点对比：
- 内置评估框架（trajectory / output / LLM-as-judge）
- 标准 benchmark 支持（Spider / BIRD 分数）
- 回归测试自动化（cassette 录放 / CI 集成）
- 测试覆盖率
- 关键差异：Vanna 四维评估器 vs Dataherald 双评估器 vs OpenChatBI RunLedger cassette

### 第 4 章：综合评估矩阵
- 6×7 打分表（每个项目每个维度 0-5 分）
- 总分（7 维度等权求和，不做加权，保持评估透明）
- 各项目一句话 harness 成熟度总结
- 光谱定位总结："GUI 工具 → 单轮生成 → Agent Pipeline → 完整 Harness"

### 第 5 章：OpenChatBI 改进路线图
按优先级分三层：

**P0（关键短板，建议立即改进）**
- 针对 OpenChatBI 在 7 维度中得分最低的 1-2 个维度
- 给出具体改进建议 + 借鉴来源

**P1（竞争力提升，建议近期改进）**
- 得分中等但有明确最佳实践可参照的维度
- 给出具体改进建议 + 借鉴来源

**P2（差异化优势，建议长期建设）**
- 已有基础但可进一步打磨成亮点的维度
- 给出具体改进建议 + 借鉴来源

### 第 6 章：附录
- 各项目 GitHub 链接与关键数据快照（截至 2026-06-08）
- 评估方法论详细说明
- 术语表

## 交付物

- 完整 Markdown 报告文件：`docs/research/data-agent-harness-comparison-2026-06-08.md`
- 预计篇幅：8,000-12,000 字（中文）
- 内含对比表格、评分矩阵、改进路线图

## 研究方法与局限

- **方法**：基于 GitHub 源码阅读、官方文档、公开论文/博客的静态分析。各项目通过 Repomix 打包或 subagent 源码调研获取实现细节。
- **局限**：
  - 未实际部署运行各项目，无运行时性能数据
  - 评分含主观判断，尽量基于可验证的代码证据
  - GitHub 数据为 2026-06-08 快照，项目持续演进中
  - 部分项目（Chat2DB Pro、MindsDB Cloud）的商业版功能未纳入评估
