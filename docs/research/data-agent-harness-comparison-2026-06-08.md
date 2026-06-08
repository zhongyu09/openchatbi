# 开源 Data Agent 项目 Harness 架构深度对比研究

> 研究日期：2026-06-08 | 研究方法：源码静态分析 + GitHub 数据 + 公开文档  
> 评估框架：Anatomy of an AI Harness 七维度模型

---

## 摘要

本研究以《Anatomy of an AI Harness》提出的七维度评估框架为标尺，对 6 个社区热度和成熟度较高的开源 Data Agent / NL-to-SQL 项目进行系统性对比分析。研究覆盖 Tool Orchestration、Error Handling、Sandbox & Security、Observability、Human-in-the-Loop、Memory & State、Eval & Testing 七个核心维度，基于源码级的架构审阅给出量化评分，并为 OpenChatBI 项目生成分优先级的改进路线图。

**核心发现**：在 Data Agent 领域，"高 GitHub Stars"与"Harness 成熟度"之间存在显著错位。社区热度最高的 MindsDB（39k stars）和 Chat2DB（25k stars）分别代表了"平台级 Harness"和"零 Harness GUI 工具"两个极端。真正在 Harness 维度形成差异化竞争力的，是 DB-GPT 的四层沙箱 + 认知记忆、Dataherald 的 golden SQL 飞轮 + confidence 门控，以及 OpenChatBI 的 LangGraph interrupt HITL 机制。

---

## 第 1 章 研究概述

### 1.1 研究背景

Data Agent——能够理解自然语言并自主操作数据库的 AI 系统——正在从实验室走向生产环境。然而，**从 "能跑通 demo" 到 "能上生产" 之间的鸿沟，本质上是 Agent Harness（运行时治理框架）的成熟度差距。**

一个生产级 Data Agent 不仅需要准确地将自然语言转为 SQL，还需要：在工具调用失败时自动恢复（Error Handling）、在执行危险操作前征求人工确认（HITL）、在长对话中管理上下文不溢出（Memory）、在代码执行时隔离风险（Sandbox）、在运行时提供可观测的追踪链路（Observability），以及在迭代 prompt 时有自动化的质量回归手段（Eval）。

本研究从 Harness 视角——而非单纯的 NL-to-SQL 准确率——对主流开源 Data Agent 项目进行系统性评估，旨在回答：**各项目在 Agent 治理维度的实现深度如何？OpenChatBI 应该从哪些项目借鉴什么？**

### 1.2 评估框架

沿用《Anatomy of an AI Harness》的 7 个核心维度，每个维度采用 0-5 分制量化评分：

| 分值 | 含义 |
|------|------|
| 0 | 无实现 |
| 1 | 最低限度 / 空壳接口 |
| 2 | 基础实现，有明显缺陷 |
| 3 | 功能完整，可用于非关键场景 |
| 4 | 生产级，覆盖主要边界情况 |
| 5 | 业界标杆，可作为最佳实践参考 |

### 1.3 研究方法与局限

**方法**：通过 Repomix 打包和 subagent 源码调研，对每个项目进行代码级架构分析，辅以官方文档、公开论文和 GitHub 数据审阅。

**局限**：
- 未实际部署运行各项目，无运行时性能数据
- 评分含主观判断，尽量基于可验证的代码证据
- GitHub 数据为 2026-06-08 快照，项目持续演进中
- 部分项目的商业版功能（Chat2DB Pro、MindsDB Cloud）未纳入评估

---

## 第 2 章 项目全景对比

### 2.1 社区指标

| 项目 | Stars | Forks | 贡献者 | 最后更新 | Release 节奏 | License |
|------|-------|-------|--------|----------|-------------|---------|
| **OpenChatBI** | — | — | — | 活跃开发中 | Beta | MIT |
| **DB-GPT** | 18,937 | 2,733 | 172 | 2026-06-08 | 每 1-3 月 | MIT |
| **Vanna.ai** | 23,577 | 2,416 | 30 | 2026-02-02 | 不规律（v2.0 大重写） | MIT |
| **Chat2DB** | ~25,400 | ~2,770 | 50 | 2026-03-22 | 开源版放缓，Pro 持续 | Apache 2.0 + 补充协议 |
| **Dataherald** | ~3,635 | 264 | 18 | **2024-07-24** | **已停滞 >23 个月** | Apache 2.0 |
| **MindsDB** | 39,265 | 6,209 | 30+ | 2026-05-25 | 持续 | 自定义 |

**关键观察**：
- MindsDB 在 star 数上遥遥领先（39k），但其 NL2SQL 只是众多能力之一
- Dataherald 虽然架构设计精良，但已实质性停止维护，依赖的 LangChain 旧版 API 面临兼容性风险
- Vanna.ai 的核心开发者贡献了 92% 的代码（bus factor = 1），项目可持续性存在风险
- DB-GPT 是唯一拥有 172 位贡献者且保持活跃发版的项目，社区健康度最高

### 2.2 技术栈矩阵

| 项目 | 语言 | Agent 框架 | LLM 支持 | 数据库支持 |
|------|------|-----------|----------|-----------|
| **OpenChatBI** | Python | LangGraph | OpenAI, Anthropic | Presto, SQLite, Trino (SQLAlchemy) |
| **DB-GPT** | Python | AWEL (自研) | 12+ (含本地部署) | 10+ (含 Excel) |
| **Vanna.ai** | Python | 自研 v2.0 | 14+ (最广) | 12+ |
| **Chat2DB** | Java/TS | 无 (非 Agent) | 10+ (含中国本土) | 16+ (最广) |
| **Dataherald** | Python | LangChain (旧版) | OpenAI 为主 | PostgreSQL 为主 |
| **MindsDB** | Python | Pydantic AI | 10+ | 33+ handler (最广) |

### 2.3 架构定位光谱

从 "LLM 辅助工具" 到 "自主 Agent 平台"，6 个项目落在不同位置：

```
GUI 工具          单轮 LLM 生成       Agent Pipeline       完整 Harness 平台
    |                  |                    |                      |
 Chat2DB          SQLChat(参考)      OpenChatBI              DB-GPT
                                     Vanna.ai v2.0          MindsDB
                                     Dataherald
```

- **Chat2DB** 本质是 AI 辅助的数据库 GUI 客户端，LLM 集成停留在 "prompt-in, SQL-out" 的单轮生成层面
- **OpenChatBI / Vanna.ai / Dataherald** 都有真正的 agent loop（多步推理、工具调用、自纠正），但 harness 完整度各有侧重
- **DB-GPT / MindsDB** 是平台级系统，不仅有 agent 能力，还内建了编排引擎、沙箱、可观测性等基础设施

---

## 第 3 章 Harness 七维度深度分析

### 3.1 Tool Orchestration — 工具编排与多步推理

> **评什么**：Agent 是否具备工具路由、多步规划、plan-execute-reflect 循环，以及工具访问权限控制的能力。这是 Agent 与"LLM wrapper"的根本分界线。

#### 横向对比

| 项目 | Agent Loop | 路由机制 | 最大迭代 | 工具权限控制 | 子图/子 Agent |
|------|-----------|----------|---------|-------------|--------------|
| **OpenChatBI** | LangGraph StateGraph | ToolNode 条件路由 | SQL retry 3 次 | 无显式 gate | Text2SQL 子图 + 数据分析子 Agent |
| **DB-GPT** | AWEL DAG + ReActAgent | BranchOperator 条件路由 | 可配置 | Bind-and-Build 资源绑定 | Planning Agent + Data Agent 协作 |
| **Vanna.ai** | LLM tool-calling loop | LLM native function calling | 10 轮 | group-based access_groups | 无 |
| **Chat2DB** | **无** | 无 | 无 | 无 | 无（MCP Server 暴露为被调用端） |
| **Dataherald** | LangChain ZeroShotAgent | LLM 自主选择工具 | 15 步 | 无 can_use_tool | RAG Agent + Finetuning Agent 双模式 |
| **MindsDB** | Pydantic AI agent loop | 结构化输出路由 | MAX_RETRIES=3 | 表级权限 _check_permissions | exploratory → final query 计划模式 |

#### 深度分析

**OpenChatBI** 的双层图架构是其编排层的核心设计亮点。主 Agent Graph 处理用户意图识别和工具分发，Text2SQL SubGraph 专注 SQL 生成流水线（schema linking → extraction → generation → execution → visualization），数据分析 Agent 基于 `deepagents` 库构建为独立子图。这种分层设计实现了关注点分离——主图不需要理解 SQL 生成的内部步骤，子图不需要处理多轮对话路由。`Send` 机制支持从主图动态派发任务到子 Agent。

**DB-GPT** 的 AWEL（Agentic Workflow Expression Language）是最重的编排方案。它本质上是一个 AI-native 的 DAG 引擎：Operator 通过 `>>` 语法链接，支持 `BranchOperator`（条件路由）、`JoinOperator`（多输入聚合）和 fan-out 模式。DAG 定义可持久化到元数据库并通过可视化 Canvas 编辑，这使得非开发人员也能编排工作流。但代价是**学习曲线陡峭**——相比 LangGraph 的 Python-native 图定义，AWEL 引入了额外的抽象层。

**Vanna.ai v2.0** 从 mixin 模式（`class MyVanna(ChromaDB, OpenAI)`）跃迁到 Agent 框架，使用 LLM native tool-calling 驱动循环。内置工具包括 `run_sql`、`visualize_data`、`run_python_file`、`pip_install` 等。其**亮点是 `search_saved_correct_tool_uses` 工具**——LLM 每次先检索历史成功的 tool-usage pattern，再决定操作，形成了"记忆驱动的编排"模式。但缺少显式的 planning step，复杂查询的分解完全依赖 LLM 隐式推理。

**Dataherald** 的双 Agent 模式（RAG Agent + Finetuning Agent）是务实的设计：冷启动时用 RAG Agent（7 个工具），积累 golden SQL 后切换到 Finetuning Agent（额外的 `GenerateSQL` 工具调用 fine-tuned 模型）。Agent 的工具链设计精良——`InfoRelevantColumns`、`ColumnEntityChecker` 等工具将计算密集的操作（embedding 相似度、entity 模糊匹配）从 LLM 中剥离出来，LLM 只需处理少量 token。

**MindsDB** 的 plan-execute 模式（exploratory query → final query）是独特的设计。Agent 先通过探索性查询（`SHOW TABLES`、`DESCRIBE`、采样查询）理解数据结构，再生成最终 SQL。这种"先看再写"的模式比"直接生成"更健壮，特别适合大 schema 场景。

#### 最佳实践

- **分层图架构**（OpenChatBI）：主图处理意图路由，子图处理领域逻辑，互不侵入
- **计算从 LLM 剥离**（Dataherald）：embedding 相似度、entity 匹配等操作放在工具内部，LLM 只做决策
- **记忆驱动编排**（Vanna.ai）：先检索历史成功 pattern 再行动，减少重复试错
- **先探索再生成**（MindsDB）：exploratory query 建立数据认知后再生成最终 SQL

---

### 3.2 Error Handling — 错误处理与自愈能力

> **评什么**：SQL 执行失败、LLM 调用异常、工具超时等场景下的恢复能力。包括错误分类粒度、重试策略和优雅降级。

#### 横向对比

| 项目 | SQL 自动重试 | 错误分类 | LLM 错误恢复 | 优雅降级 |
|------|-------------|----------|-------------|---------|
| **OpenChatBI** | 3 次，带错误分类 | syntax/permission/timeout | call_llm_with_retry | 超时/重试耗尽后结束 |
| **DB-GPT** | AWEL ON ERROR RETRY N | SQL 安全清洗分类 | JSON recovery 容错 | ON ERROR LOG 继续 |
| **Vanna.ai** | **无**（依赖 LLM 自发） | 粗粒度 error_type 标记 | ErrorRecoveryStrategy（空壳） | 默认 FAIL |
| **Chat2DB** | 无 | 无 | SSE 流错误清理 | 无 |
| **Dataherald** | Agent 15 步自纠正 | 结构化 error code | catch_exceptions 统一处理 | early_stopping 生成最终答案 |
| **MindsDB** | MAX_RETRIES=3 | QueryError 区分 | 累积错误上下文 | null-filled DataFrame |

#### 深度分析

**OpenChatBI** 的 SQL 重试机制是框架级的强制行为——`generate_sql` 节点在 SQL 执行失败后最多重试 3 次，并将错误信息反馈给 LLM 重新生成。`text2sql_utils.py` 中的错误分类区分了语法错误、权限错误和超时，使 LLM 能针对性地修正。这是中规中矩但可靠的设计。

**Dataherald** 的错误处理最为成熟。Agent 拥有 15 步的迭代空间（`AGENT_MAX_ITERATIONS=15`），在 SQL 执行失败时，agent 会自动查看错误信息、检查 schema、修改 SQL 并重试。`early_stopping_method="generate"` 确保即使达到迭代上限，agent 仍会尝试给出最终答案而非直接失败。`catch_exceptions()` 装饰器统一处理 OpenAI/Google/SQLAlchemy 异常。SQL 提取也有容错——先从 markdown code block 提取，失败则从 intermediate steps 中找最后一个 SELECT。

**DB-GPT** 的 AWEL DSL 提供了声明式的错误处理语义：`ON ERROR FAIL`（快速失败）、`ON ERROR RETRY N TIMES`（自动重试）、`ON ERROR LOG`（记录并继续）。这种"在编排层定义错误策略"的设计比在每个工具内部硬编码 try-catch 更灵活。JSON recovery（`find_json_objects`）从嘈杂的 LLM 输出中提取有效 JSON 也是实用的容错手段。

**Vanna.ai** 的错误恢复是最大的空洞。`ErrorRecoveryStrategy` 定义了完善的接口（`handle_tool_error` / `handle_llm_error`，支持 RETRY / FAIL / SKIP），但默认实现直接返回 FAIL。SQL 执行失败后，错误信息虽然会回传给 LLM（LLM 可以选择重新生成），但这完全是 LLM 的自发行为，框架层面没有任何强制重试逻辑。

**MindsDB** 的亮点是**累积错误上下文**——Chart Agent 在重试时不仅传递当前错误信息，还累积之前所有失败的错误上下文，帮助 LLM 避免重复犯同一个错误。`null-filled DataFrame` 优雅降级确保下游流程不会因为单个查询失败而完全中断。

#### 最佳实践

- **框架级强制重试 + 错误分类**（OpenChatBI / Dataherald）：不依赖 LLM 自发行为，在编排层保证重试发生
- **声明式错误策略**（DB-GPT AWEL）：在 DAG 定义层控制 retry/fail/log，而非硬编码在业务逻辑中
- **累积错误上下文**（MindsDB）：多轮重试时向 LLM 传递历史错误，避免重复犯错
- **SQL 提取容错**（Dataherald）：多种模式匹配 SQL 输出，不因格式问题丢失有效 SQL

---

### 3.3 Sandbox & Security — 沙箱隔离与安全防线

> **评什么**：代码执行隔离等级、SQL 注入防护机制、权限控制粒度、已知安全漏洞。Data Agent 直接操作数据库，安全是生产部署的硬性门槛。

#### 横向对比

| 项目 | SQL 注入防护 | 代码执行沙箱 | 权限控制 | 已知漏洞 |
|------|-------------|-------------|---------|---------|
| **OpenChatBI** | 无专门防护 | RestrictedPython / Docker / Local 三级 | 无用户级 | 无公开 |
| **DB-GPT** | SQL 白名单（SELECT/INSERT/UPDATE/DELETE/ALTER/CREATE TABLE） | 四层隔离（lyric worker + Docker + 资源监控） | 资源级绑定 | 无公开 |
| **Vanna.ai** | **无** | **无**（直接 sys.executable 执行 Python） | group-based（UI 层面） | 无公开 |
| **Chat2DB** | **无** | **无** | Pro 版 SQL 审计 | **CVSS 9.8**（硬编码凭据 + RCE + SQL 注入） |
| **Dataherald** | sqlparse 关键词黑名单 | 无代码执行 | Fernet 加密 URI + Enterprise RBAC | 无公开 |
| **MindsDB** | **AST 级查询白名单** | 无独立沙箱（SQL only） | 表级权限 + 多租户 + 标识符转义 | 无公开 |

#### 深度分析

**DB-GPT** 的四层沙箱隔离模型是所有项目中最完善的：User Layer（请求提交）→ Control Layer（生命周期、安全策略、资源分配）→ Execution Layer（lyric worker 进程隔离）→ Display Layer（格式化返回）。Python 和 JS/TS 代码在独立进程（`lyric-py-worker` / `lyric-js-worker`）中执行，不在主 webserver 进程内。`psutil` 跟踪 CPU/内存消耗防止资源耗尽攻击，Docker 后端提供额外容器级隔离。SQL 层面有白名单机制，阻止 `SYSTEM`、`GRANT`、`DROP DATABASE`、`INSTALL`、`EXPORT` 等危险操作。

**MindsDB** 的 AST 级查询安全是 SQL 防护的标杆做法。`_check_permissions` 通过 SQL AST 解析（而非字符串匹配）校验查询类型，仅允许 `SELECT`、`SHOW`、`DESCRIBE`、`EXPLAIN`、`UNION` 等只读操作。自动添加 `LIMIT 100` 防止全表扫描。标识符转义（quoting）防止 SQL 注入。CTE 引用独立校验确保不被用于提权。这种 AST 级别的防护远比关键词黑名单可靠。

**OpenChatBI** 的三级代码执行沙箱是差异化设计：RestrictedPython（白名单内置函数）→ LocalExecutor（开发环境）→ DockerExecutor（生产环境）。RestrictedPython 通过 `compile_restricted` 限制代码能力，配合 `safe_builtins` 和 `safer_getattr` 阻止危险操作。但 SQL 层面缺少专门的注入防护——生成的 SQL 直接通过 SQLAlchemy 执行，没有 AST 级校验或操作白名单。

**Vanna.ai** 的安全是所有项目中最薄弱的。`RunPythonFileTool` 直接用 `sys.executable` 执行 Python 文件，`PipInstallTool` 直接调 `pip install`——没有沙箱、没有容器隔离、没有代码审查。SQL 执行同样没有任何防护。`AuditLogger` 的参数脱敏是唯一的安全亮点，但这是事后审计而非事前防护。

**Chat2DB** 是安全问题最严重的项目。2026 年 5 月披露的漏洞链包括：默认硬编码管理员凭据 `chat2db:chat2db`、通过 H2 RUNSCRIPT 的 JDBC URL 远程代码执行（RCE）、任意 JAR 上传，综合 CVSS 评分 9.8。DM 数据库导出模块有 8 处 `String.format()` SQL 拼接注入。这些漏洞表明其安全设计存在系统性缺失。

**Dataherald** 采用 sqlparse 关键词黑名单过滤 `DROP`、`DELETE`、`UPDATE`、`INSERT` 等操作，只允许 SELECT 类查询通过。这比无防护好，但不如 AST 级分析可靠——精心构造的 SQL（如注释中嵌入关键词）可能绕过。连接安全方面做得好：Fernet 对称加密存储所有 DB 连接 URI，支持 SSH 隧道。

#### 最佳实践

- **AST 级 SQL 校验**（MindsDB）：基于语法树分析操作类型，而非字符串匹配，是最可靠的 SQL 防护方式
- **多层纵深防御**（DB-GPT）：进程隔离 + 资源监控 + Docker + SQL 白名单，任一层被突破不影响整体安全
- **三级沙箱分级**（OpenChatBI）：根据部署环境选择合适的隔离级别，平衡安全与开发便利

---

### 3.4 Observability — 可观测性与运维支撑

> **评什么**：分布式追踪、LLM 指标（token 消耗、延迟、成本）、审计日志、运维工具链。Agent 系统的调试复杂度远高于传统应用，可观测性决定了"出了问题能不能定位"。

#### 横向对比

| 项目 | 分布式追踪 | LLM Metrics | 审计日志 | 运维工具 |
|------|-----------|-------------|---------|---------|
| **OpenChatBI** | 无 | 无 | 无 | Python logging |
| **DB-GPT** | 内建 span-based + OTel 导出 | TTFT / token 吞吐量 / prefill 速度 | 结构化 JSON | `dbgpt trace` CLI |
| **Vanna.ai** | ObservabilityProvider 接口（无默认实现） | 无 | AuditLogger（参数脱敏） | 无 |
| **Chat2DB** | 无 | 无 | Pro 版 SQL 审计 | Spring Boot 标准日志 |
| **Dataherald** | LangSmith 集成 | token 计量（OpenAI callback） | Intermediate steps（隐私保护） | Admin Console GUI |
| **MindsDB** | Langfuse decorator + OTel collector | 无专项 | 无专项 | Prometheus metrics（查询耗时/行数/延迟） |

#### 深度分析

**DB-GPT** 的可观测性是从设计之初就内建的，而非事后补丁。span-based 追踪系统在启动时初始化 root tracer，支持 `SpanType` 分类的层次化 span 关系，可导出到 OpenTelemetry 企业级监控栈。`LLMPerformanceMonitor` 自动追踪 `input_token_count`、TTFT（首 token 时间）、prefill/decode 速度和吞吐量。`dbgpt trace` CLI 工具允许在终端直接查询和可视化追踪数据。这种"从自然语言输入到最终响应"的全链路追踪覆盖了 AWEL operator、multi-agent 交互和 LLM 调用。

**MindsDB** 采用三件套方案：Langfuse 做 LLM trace（`@langfuse_traced_stream()` 装饰器，含 trace_id/span/metadata）、Prometheus 做业务指标（查询耗时 Summary、响应行数 Summary、REST 延迟 Histogram）、OpenTelemetry collector 做分布式追踪。三者条件启用以避免性能开销。

**Vanna.ai** 的设计最具启发性但"无电池"。`ObservabilityProvider` 抽象支持 `record_metric()` 和 `create_span()` / `end_span()`，几乎每个关键路径都有 span 埋点——user resolution、conversation load、system prompt build、tool execution、hook 执行、context enrichment。但**所有方法的默认实现都是空 `pass`**，需要用户自行对接 Prometheus/Datadog/OpenTelemetry。

**Dataherald** 的 LangSmith 集成是最轻量的方案——设置环境变量即可获得完整的 agent tool-calling trace。每次请求记录 `tokens_used`（通过 `get_openai_callback`）。`intermediate_steps` 中的查询结果被替换为 "QUERY RESULTS ARE NOT STORED FOR PRIVACY REASONS"，这种隐私保护设计值得注意。

**OpenChatBI** 当前仅有 Python logging，是 7 个维度中得分最低的。没有 LLM 调用追踪、没有 token/cost 计量、没有分布式追踪基础设施。对于一个双层图架构的系统，这意味着当 Text2SQL 子图的 schema linking 选错了表，或者数据分析子 Agent 的异常检测给出了误报，开发者几乎无法从日志中定位问题根因。

#### 最佳实践

- **内建 span tracing + OTel 导出**（DB-GPT）：在框架级埋点，不依赖外部集成
- **Langfuse + Prometheus + OTel 三件套**（MindsDB）：LLM trace、业务指标、分布式追踪各司其职
- **全面接口 + 零默认实现**（Vanna.ai）：设计参考但缺少开箱即用
- **隐私保护的 intermediate steps**（Dataherald）：记录 agent 推理过程但不持久化查询结果

---

### 3.5 Human-in-the-Loop — 人机协作与审批机制

> **评什么**：Agent 在执行高风险操作前是否征求人工确认、是否支持需求澄清对话、是否有反馈闭环让人工纠正结果。这是"自主 Agent"与"受控 Agent"的分水岭。

#### 横向对比

| 项目 | 审批门控 | 需求澄清 | 反馈闭环 | 置信度评估 |
|------|---------|---------|---------|-----------|
| **OpenChatBI** | LangGraph `interrupt` | AskHuman 工具 | 无 golden SQL | 无 |
| **DB-GPT** | 无显式审批 | 对话式交互 | 知识库管理 | correctness_check |
| **Vanna.ai** | **无** | **无** | save_question_tool_args | 无 |
| **Chat2DB** | 天然 HITL（手动执行） | 无 | 无 | 无 |
| **Dataherald** | Confidence threshold 门控 | 无 | **Golden SQL 飞轮** | SimpleEvaluator 0-1 分 |
| **MindsDB** | 无（Agent 自主执行） | 无 | Knowledge Base | 无 |

#### 深度分析

**OpenChatBI** 在 HITL 维度是所有真正 Agent 项目中最强的。LangGraph 的 `interrupt` 机制允许在 SQL 生成流水线的任意节点暂停执行、等待人工输入后继续。`AskHuman` 工具让 Agent 能主动向用户提问以澄清需求（"你说的'上周'是指自然周还是最近 7 天？"），支持选项按钮式的交互。这种设计将 Agent 从"自主执行"模式切换为"提议-确认-执行"模式，在生产环境中至关重要。

**Dataherald** 的 HITL 设计围绕**反馈闭环**展开，是最具生产意识的方案。Confidence threshold 门控允许设置阈值（如 0.7），低于阈值的 SQL 结果会被 block 住等待人工审核。核心飞轮机制是 golden SQL：用户可将 agent 生成的 SQL 标记为 verified，加入向量数据库，后续自动做 few-shot retrieval。这意味着**系统使用越多越准确**——每一次人工纠正都在增强 agent 的能力。Instructions 系统允许在数据库/表/列级别添加业务约束规则。

**Chat2DB** 的 HITL 是"架构简单的副产品"而非设计性安全机制——LLM 只生成 SQL 文本，用户必须手动点击执行。这确实阻止了 agent 自主执行危险 SQL，但也牺牲了自动化能力。

**DB-GPT** 和 **MindsDB** 都缺少显式的审批门控。DB-GPT 的 ReActAgent 自主执行操作链，没有类似 LangGraph `interrupt` 的中断点。MindsDB 的 agent 同样自主执行 SQL。两者都是"人发起 → AI 执行 → 人查看结果"的模式，而非"AI 提议 → 人审批 → AI 执行"。

**Vanna.ai** 在 HITL 维度最为薄弱。没有 confirmation gate、没有 clarification flow、SQL 生成后直接执行。`save_question_tool_args` 工具保存成功的 pattern 是被动反馈，不是主动的人工纠正机制。

#### 最佳实践

- **图级 interrupt 机制**（OpenChatBI）：在编排图的任意节点可暂停/恢复，最灵活的 HITL 实现
- **Confidence 门控 + golden SQL 飞轮**（Dataherald）："低置信度阻断 + 人工验证 → 增强训练数据"形成正向循环
- **主动需求澄清**（OpenChatBI AskHuman）：agent 能主动提问，而非被动猜测用户意图

---

### 3.6 Memory & State — 记忆系统与状态管理

> **评什么**：对话持久化、长期记忆（跨会话的知识积累）、上下文管理（防止 token 溢出）、多租户隔离能力。

#### 横向对比

| 项目 | 对话持久化 | 长期记忆 | 上下文管理 | 多租户 |
|------|-----------|---------|-----------|--------|
| **OpenChatBI** | SQLite checkpoint | langmem 向量存储 | ContextManager 摘要压缩 | 无 |
| **DB-GPT** | GptsMemory conv_id | 三层认知记忆 + 知识图谱 | ContextManager token 预算 | session_id 绑定 |
| **Vanna.ai** | ConversationStore 抽象（默认 in-memory） | AgentMemory（tool-usage pattern） | ConversationFilter 截断 + LlmMiddleware | 无 |
| **Chat2DB** | LocalCache 进程内 + localStorage | Milvus/ES schema 索引 | 无 | 无 |
| **Dataherald** | MongoDB | Vector DB（golden SQL + schema） | Smart Cache | 多组织隔离 |
| **MindsDB** | 会话级 message_history | Knowledge Base 语义记忆 | 无显式管理 | company_id / user_id 隔离 |

#### 深度分析

**DB-GPT** 的三层认知记忆模型是所有项目中最复杂也最有深度的设计，仿照人类记忆分为：Sensory Memory（临时缓冲，超容量丢弃）→ Short-Term Memory（持有当前上下文，检索历史增强）→ Long-Term Memory（向量存储持久化）。`ImportanceScorer` 用 LLM 评估每个信息片段的重要性分数，`LLMInsightExtractor` 提取抽象洞察。`TimeWeightedEmbeddingRetriever` 时间加权嵌入检索避免旧信息主导。此外还支持知识图谱（Neo4j/TuGraph），实现 GraphRAG。

**Vanna.ai** 的 AgentMemory 设计虽然简单但独到。核心创新是**将 "question → tool → args" 三元组作为记忆单位**，而非传统 RAG 中的 "question → SQL" 对。这意味着 agent 不仅记住了"哪个问题用哪个 SQL"，还记住了"哪个问题应该先调哪个工具、传什么参数"。`LlmContextEnhancer` 在系统提示中自动注入相关 text memories，`ConversationFilter` 可过滤/截断历史消息避免 context window 超限。

**OpenChatBI** 的记忆系统基于 `langmem` 库，支持 `manage_memory_tool` 和 `search_memory_tool` 两个工具。SQLite checkpoint 提供对话持久化，`ContextManager` 在对话过长时自动摘要压缩历史消息。这是务实且完整的方案，但缺少 DB-GPT 那样的重要性评分和时间衰减机制。

**Dataherald** 的记忆聚焦于"训练数据积累"：golden SQL + schema descriptions + column metadata + business instructions 存入 MongoDB 和 Vector DB。`retrieve_context_for_question()` 基于 embedding 相似度检索相关上下文。但它是**无状态的单轮 Q&A**设计，没有多轮对话上下文追踪。

#### 最佳实践

- **三层认知记忆 + 重要性评分**（DB-GPT）：结构化的记忆管理，关键信息不被噪声淹没
- **Tool-usage pattern 记忆**（Vanna.ai）：记住"怎么做"而非只记住"做了什么"
- **摘要压缩 + 长期记忆双轨**（OpenChatBI）：短期靠压缩保鲜，长期靠向量检索
- **多租户隔离**（MindsDB / Dataherald）：生产环境的基本要求

---

### 3.7 Eval & Testing — 评估体系与质量保障

> **评什么**：是否有内置的评估框架、标准 benchmark 支持、回归测试自动化能力。这是"改一次 prompt 是否需要人工全量回归"的分水岭。

#### 横向对比

| 项目 | 内置 Eval 框架 | 标准 Benchmark | 回归自动化 | 测试覆盖 |
|------|---------------|---------------|-----------|---------|
| **OpenChatBI** | RunLedger cassette 录放 | 无公开分数 | CI 可回放 | 409 个测试函数 |
| **DB-GPT** | BenchmarkService（EXACT/CONTAIN/JSON_PATH） | DB-GPT-Hub Spider 微调 | 结果持久化 Excel | 未公开覆盖率 |
| **Vanna.ai** | **四维评估器** + ComparisonReport | 无公开分数 | EvaluationRunner 并行 | 14 个测试文件 |
| **Chat2DB** | 无 | Chat2DB-SQL-7B Spider 77.3% | 无 | 520 open issues |
| **Dataherald** | 双评估器（Simple + Agent） | DIN-SQL Spider/BIRD 顶级 | 无可复现脚本 | 模块级测试 |
| **MindsDB** | Knowledge Base eval（MRR/nDCG/Hit@k） | 无 NL2SQL 专项 | 自动生成测试 Q&A | 无公开覆盖率 |

#### 深度分析

**Vanna.ai** 的评估框架设计最为完整。四维评估器覆盖：`TrajectoryEvaluator`（验证工具调用序列正确性）、`OutputEvaluator`（验证输出内容）、`LLMAsJudgeEvaluator`（用另一个 LLM 评判质量）、`EfficiencyEvaluator`（验证执行时间和 token 消耗）。`EvaluationRunner` 支持并行执行，`ComparisonReport` 可对比不同 LLM/配置的表现差异。这是接近工业级评估体系的设计。

**OpenChatBI** 的 RunLedger cassette 录放机制是独特的 CI 友好方案。录制模式下连接真实环境记录工具输出，回放模式下用 JSONL cassette 代替真实调用，实现确定性 CI 回归测试。409 个测试函数覆盖了 Text2SQL 全流程、异常检测算法、Adtributor 归因、上下文管理和 graph state 操作。但当前仅有 1 个评估用例（t1），需要扩充。

**Dataherald** 的双评估器方案在速度和深度之间提供了选择：SimpleEvaluator（几秒，LLM 逐项检查常见错误）和 EvaluationAgent（40-50 秒，实际执行 SQL 验证）。6 步深度检查包括 SELECT 列对应性、WHERE 条件正确性、计算检查、子查询分解、JOIN 列匹配和执行结果检查。但评估器自身的准确率没有 ground truth 验证。

**DB-GPT** 的 BenchmarkService 提供三种比较策略（EXACT_MATCH / CONTAIN_MATCH / JSON_PATH），结果分类为 RIGHT/WRONG/FAILED/EXCEPTION 并持久化到 Excel。DB-GPT-Hub 使用 Spider 数据集做模型微调评估。但缺少执行准确率（EX）等 Text2SQL 标准评估指标。

**MindsDB** 的 Knowledge Base eval 最专业，包含 MRR、nDCG、Hit@k、Precision@k、Recall 曲线、延迟等完整指标体系，支持自动生成测试 Q&A 对。但这是 RAG 检索评估，NL2SQL 本身没有独立 eval 体系。

#### 最佳实践

- **四维评估器 + 多配置对比**（Vanna.ai）：trajectory、output、LLM-as-judge、efficiency 全覆盖
- **Cassette 录放 + CI 集成**（OpenChatBI RunLedger）：确定性回归测试，不依赖外部服务
- **速度/深度可选双评估器**（Dataherald）：日常快速检查 + 定期深度验证

---

## 第 4 章 综合评估矩阵

### 4.1 打分表

| 项目 | Tool Orch | Error | Security | Observ | HITL | Memory | Eval | **总分** |
|------|-----------|-------|----------|--------|------|--------|------|----------|
| **OpenChatBI** | 4 | 3 | 3 | 1 | 4 | 4 | 3 | **22/35** |
| **DB-GPT** | 4 | 3 | 4 | 4 | 2 | 5 | 2 | **24/35** |
| **Vanna.ai** | 3 | 1 | 1 | 2 | 1 | 4 | 3 | **15/35** |
| **Chat2DB** | 0 | 1 | 0 | 1 | 2 | 2 | 1 | **7/35** |
| **Dataherald** | 3 | 4 | 3 | 3 | 4 | 2 | 3 | **22/35** |
| **MindsDB** | 4 | 3 | 4 | 4 | 2 | 3 | 2 | **22/35** |

### 4.2 评分依据说明

**OpenChatBI (22/35)**
- **亮点**：HITL（LangGraph interrupt + AskHuman 是 Agent 项目中最完整的 HITL 实现）、Memory（langmem + ContextManager 摘要压缩）、Tool Orchestration（双层图架构清晰）
- **短板**：Observability 是唯一的 1 分，仅有 Python logging，无追踪、无 metrics、无审计

**DB-GPT (24/35) — 总分最高**
- **亮点**：Memory（三层认知记忆，业界独创）、Security（四层沙箱隔离）、Observability（内建 span tracing + OTel）
- **短板**：HITL（无显式审批门控）、Eval（比较策略粗粒度）

**Vanna.ai (15/35)**
- **亮点**：Memory（tool-usage pattern 记忆）、Eval（四维评估器设计完整）
- **短板**：Security（无 SQL 防护、无代码沙箱，最危险）、Error Handling（recovery 空壳）、HITL（无任何机制）

**Chat2DB (7/35) — 总分最低**
- **本质不是 Agent**，缺少 agent loop、安全沙箱、可观测性等 harness 基础设施
- **CVSS 9.8 已知漏洞**使得 Security 得分为 0
- 作为"高 star 数不等于高 harness 成熟度"的典型反例

**Dataherald (22/35)**
- **亮点**：Error Handling（15 步 agent 自纠正，最成熟）、HITL（confidence 门控 + golden SQL 飞轮）
- **短板**：Memory（无多轮对话上下文）、**项目已停滞 23+ 个月**

**MindsDB (22/35)**
- **亮点**：Security（AST 级查询白名单，最可靠）、Observability（Langfuse+Prometheus+OTel 三件套）
- **短板**：HITL（agent 自主执行无审批）、Eval（NL2SQL 无专项 eval）

### 4.3 Harness 成熟度光谱

```
零 Harness          部分 Harness          较完整 Harness         平台级 Harness
(7/35)              (15/35)               (22/35)                (24/35)
   |                   |                      |                      |
Chat2DB            Vanna.ai            OpenChatBI              DB-GPT
                                       Dataherald
                                       MindsDB
```

**关键洞察**：

1. **三个 22 分项目的差异化方向完全不同**：OpenChatBI 强在 HITL，Dataherald 强在 Error Handling，MindsDB 强在 Security + Observability。这意味着改进方向应该"扬长补短"而非追求全面平庸。

2. **Star 数与 Harness 成熟度严重错位**：最高 star（MindsDB 39k）不是最高分（DB-GPT 24 分），最低分（Chat2DB 7 分）却有 25k stars。社区热度反映的是"使用门槛"和"功能覆盖面"，而非"Agent 治理深度"。

3. **Dataherald 虽然得分高但已停滞**，其设计理念（golden SQL 飞轮、confidence 门控）值得借鉴，但代码已不可直接依赖。

---

## 第 5 章 OpenChatBI 改进路线图

基于评估矩阵，OpenChatBI 当前总分 22/35，与 Dataherald、MindsDB 并列，但维度分布不均——HITL(4) 和 Memory(4) 是亮点，Observability(1) 是明显短板。改进策略是"补最短板 → 强化竞争力 → 巩固差异化优势"。

### P0：可观测性 — 当前最大短板（1→3 分目标）

**现状**：仅有 Python logging，无 LLM 追踪、无 token/cost 计量、无分布式追踪。双层图架构的调试几乎是黑盒。

**改进建议**：

1. **接入 LangSmith 或 Langfuse**（借鉴 Dataherald / MindsDB）
   - 最低成本方案：设置 `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` 即可获得 LangGraph 全链路 trace
   - 自托管方案：Langfuse 开源版 + `@langfuse_traced_stream()` 装饰器
   - 覆盖范围：主 Agent Graph + Text2SQL SubGraph + 数据分析子 Agent 的每次 LLM 调用和工具执行

2. **增加 LLM token/cost metrics**（借鉴 DB-GPT `LLMPerformanceMonitor`）
   - 在 `call_llm_chat_model_with_retry` 中记录每次调用的 input/output token 数和推理延迟
   - 通过 `streaming.py` 的 `StreamEvent` 传递 metrics 到各 UI 层

3. **审计日志**（借鉴 Vanna.ai `AuditLogger`）
   - 记录每次工具调用的输入参数（敏感字段脱敏）和执行结果
   - 特别是 SQL 执行日志，对合规和安全审计至关重要

### P1：错误处理增强 + 评估扩充（3→4 分目标）

**Error Handling 改进**（借鉴 Dataherald + MindsDB）：

1. **结构化错误分类**
   - 当前 SQL retry 已有基础，但错误分类粒度可提升
   - 参考 Dataherald 的 `SQLInjectionError` / `InvalidDBConnectionError` / `EmptyDBError` 等结构化类型
   - 每种错误类型对应不同的恢复策略（retry / 换表 / 报错给用户）

2. **累积错误上下文**（借鉴 MindsDB）
   - 多轮 SQL 重试时，将之前所有失败的错误信息一并传递给 LLM
   - 避免 LLM 在第 2 次重试时犯和第 1 次相同的错误

**Eval 扩充**（借鉴 Vanna.ai + Dataherald）：

1. **扩充 RunLedger 评估用例**
   - 当前仅 1 个 case（t1），建议增加到 20+ 覆盖常见查询模式
   - 包含：聚合查询、多表 JOIN、时间范围过滤、异常检测、可视化生成

2. **增加 LLM-as-Judge 评估维度**（借鉴 Vanna.ai `LLMAsJudgeEvaluator`）
   - 用独立 LLM 评判生成 SQL 的语义正确性
   - 与 RunLedger 的确定性回放互补

### P2：差异化优势巩固（4→5 分目标）

**HITL 增强**（当前已是标杆，进一步拉开差距）：

1. **引入 Golden SQL 飞轮**（借鉴 Dataherald）
   - 用户确认 SQL 正确后存入向量数据库，后续自动做 few-shot retrieval
   - 与现有 `search_knowledge` 工具集成
   - 实现"系统使用越多越准确"的正向循环

2. **Confidence 门控**（借鉴 Dataherald `SimpleEvaluator`）
   - 对每个生成的 SQL 进行置信度评分
   - 低置信度结果触发 LangGraph interrupt 等待人工审核，而非直接返回

**Memory 增强**（当前已有良好基础）：

1. **Tool-usage pattern 记忆**（借鉴 Vanna.ai）
   - 不仅记住 Q→SQL 映射，还记住 Q→工具调用链的成功模式
   - 与 langmem 的长期记忆互补

2. **重要性评分 + 时间衰减**（借鉴 DB-GPT）
   - 对长期记忆中的条目做重要性评分，避免过时信息污染检索结果

### 改进优先级总结

| 优先级 | 维度 | 当前分 | 目标分 | 核心借鉴来源 | 预估工作量 |
|--------|------|--------|--------|-------------|-----------|
| **P0** | Observability | 1 | 3 | DB-GPT, MindsDB, Dataherald | 中（接入 LangSmith/Langfuse + metrics 埋点） |
| **P1** | Error Handling | 3 | 4 | Dataherald, MindsDB | 小（结构化错误分类 + 累积上下文） |
| **P1** | Eval | 3 | 4 | Vanna.ai, Dataherald | 中（扩充 RunLedger cases + LLM-as-Judge） |
| **P2** | HITL | 4 | 5 | Dataherald | 中（Golden SQL 飞轮 + Confidence 门控） |
| **P2** | Memory | 4 | 5 | Vanna.ai, DB-GPT | 中（Pattern 记忆 + 重要性评分） |

---

## 第 6 章 附录

### A. 项目链接与数据快照

| 项目 | GitHub 地址 | 数据截止日期 |
|------|------------|-------------|
| OpenChatBI | 内部项目 | 2026-06-08 |
| DB-GPT | github.com/eosphoros-ai/DB-GPT | 2026-06-08 |
| Vanna.ai | github.com/vanna-ai/vanna | 2026-06-08 |
| Chat2DB | github.com/CodePhiliaX/Chat2DB | 2026-06-08 |
| Dataherald | github.com/Dataherald/dataherald | 2026-06-08 |
| MindsDB | github.com/mindsdb/mindsdb | 2026-06-08 |

### B. 评估方法论

1. **源码分析**：通过 Repomix 打包本地项目（OpenChatBI）、subagent 源码调研（其余 5 个项目），获取每个项目的代码级架构细节
2. **文档审阅**：官方 README、API 文档、架构文档、公开论文
3. **GitHub 数据**：Stars、Contributors、Release 历史、Issue 数量
4. **评分原则**：
   - 评分基于公开源码中的实际实现，而非文档声明或 roadmap
   - 有接口但无默认实现（如 Vanna.ai 的 ObservabilityProvider）按接口设计质量适度给分，但低于有实际实现的项目
   - 商业版/Pro 版功能不纳入评分
   - 已停滞项目（Dataherald）按最后可用版本的代码评分，不因停滞降分

### C. 术语表

| 术语 | 说明 |
|------|------|
| Agent Harness | AI Agent 的运行时治理框架，包括工具编排、错误处理、安全隔离、可观测性、人机协作、记忆管理和评估体系 |
| HITL | Human-in-the-Loop，人机协作机制，Agent 在关键决策点引入人工判断 |
| AWEL | Agentic Workflow Expression Language，DB-GPT 的工作流编排 DSL |
| MCP | Model Context Protocol，AI 模型与外部工具交互的标准协议 |
| Golden SQL | 经人工验证的 NL-SQL 配对，用于 few-shot 检索和模型微调 |
| Cassette | 预录制的工具调用输入/输出记录，用于确定性回归测试 |
| AST | Abstract Syntax Tree，抽象语法树，用于代码/SQL 的结构化分析 |
| Schema Linking | 将自然语言查询中的实体映射到数据库 schema（表名、列名）的过程 |
| LangSmith / Langfuse | LLM 调用追踪和可观测性平台 |
| RunLedger | OpenChatBI 使用的评估框架，支持 cassette 录制和回放 |
