# OpenChatBI Harness 整体改进方案 — 设计文档

> 基于《开源 Data Agent 项目 Harness 架构深度对比研究》(2026-06-08) 第 5 章改进路线图,
> 经一轮对全部 5 个维度的 **代码级深挖 + 对抗式核查**(workflow `wf_0f971d9a-73c`,10 agent)收敛而成。
> 本文档是落地级技术设计 + 战略排期的分层方案,作为后续 implementation plan 的输入。

---

## 1. 背景与目标

OpenChatBI 当前 Harness 总分 **22/35**(与 Dataherald、MindsDB 并列),维度分布不均:HITL(4)/Memory(4) 是亮点,**Observability(1) 是最大短板**,Error/Eval(3) 中等。报告给出了 P0/P1/P2 的改进建议,但那是战略层清单。

本方案的目标:把报告的 5 项改进建议(Observability / Error Handling / Eval / HITL / Memory)落到 OpenChatBI 的**真实代码**上,组织成一个**整体方案**。这 5 个目标维度子分从 **15 提升到 21**(各维度详见 §7),其余两维(Tool Orchestration 4、Sandbox & Security 3,共 7 分)不变,故 Harness **总分从 22 提升到 28(/35)**。

### 1.1 代码级核查对报告的 3 处修正

深挖发现报告有 3 处与真实代码不符,直接影响改进范围:

1. **错误处理的"累积错误上下文"已实现**。`regenerate_sql_node`(`generate_sql.py:440-444`)已把 `previous_sql_errors` 全部历史逐条(Attempt 1..N)拼进重试 prompt。报告建议"借鉴 MindsDB 从零做"是误判——这条从"新建"缩水成"增量增强"。
2. **HITL 置信门控按报告落点其实是"先执行再审查"**。唯一可落位置在 `execute_sql` 成功**之后**,此时 SQL 已 commit 到数仓。因 `_validate_sql_safety` 强制只读 SELECT 可接受;真·执行前门控需重排节点顺序。**已决策:执行后审查为默认,执行前门控作为 config 开关**(设计见 §5.2,决策见 §8 ADR #2)。
3. **Eval 扩充有致命前提**:RunLedger 的 agent adapter **完全 stub 了 LLM**(`agent.py:_stub_llm_call`),光扩 RunLedger 用例只验证调用链管道,验证不了 SQL 质量。LLM-as-Judge 必须**与用例同时落地**且跑在 RunLedger **之外**。可复用的种子语料极薄——RunLedger 当前**仅 1 条 case**(`evals/runledger/cases/t1.yaml`,prompt=`OpenChatBI`);`tests/conftest.py:214-223` 的 2 条 COUNT/AVG 是**单测 fixture**(`sample_sql_examples`)而非 RunLedger 输入。两者都不足以派生 20+ 真实用例,gold SQL 须手写,工作量上修为 **L**。

### 1.2 核心发现:5 个维度共享 3 块地基

对抗核查的 cross-dimension 依赖揭示:5 个维度**不是独立的**。若按报告 P0→P1→P2 顺序逐个做,会把同一块东西实现两遍。

| 共享地基 | 被哪些维度复用 | 不共享的代价 |
|---|---|---|
| **S1 可观测性底座** | Observability 本体 / Error 错误指标 / HITL 评估耗时 / Eval 的 token-cost budget | 指标埋点各维度零散重做 |
| **S2 SQL 质量评估器** | HITL 置信门控 / Eval 的 LLM-as-Judge / Memory 防"成功≠正确"污染闸门 | **同一评估器实现 3 遍** |
| **S3 运行时可变"已学习 SQL 知识库"** | HITL Golden-SQL 飞轮(`source='golden'`)/ Memory 自动捕获的 SQL 模式(`source='auto'`)——**同一个 store** | `data.py:16` 静态单例改造**做两遍**、两套检索/写入/打分 |

这是本方案采用"地基优先分层"而非 5 份独立路线图的根本理由。

---

## 2. 设计原则

1. **地基优先,不重复造**:被 ≥2 个维度复用的能力,先在阶段 0 建成单一模块,阶段 1 只接线不重造。
2. **特性默认关、可灰度**:所有新增行为(置信门控、Golden-SQL、模式记忆、衰减)默认 `enabled=False`,经 config 显式开启,保证现有行为零回归。
3. **不破坏现有契约**:保留 `SQL_*` 状态码常量取值、保留测试断言耦合的人类可读 `error_type` 字符串;新结构化信息进**新字段**。
4. **密钥不入库**:Langfuse 等凭据只走环境变量 / `.env`(已 gitignore),文档与配置模板只出现变量名,不出现真值。
5. **YAGNI**:不做报告未要求、当前无消费方的能力(如跨工具统一错误体系仅列为可选)。

---

## 3. 总体架构:三段式地基优先

```
阶段 0 共享地基 ──────────→ 阶段 1 维度功能接入 ──────────→ 阶段 2 证明
  S1 可观测性底座               Obs 流量展示                      Eval 语料 1→20+
  S2 SQL 质量评估器             Error 结构化分类 + 策略路由        LLM-as-Judge(复用 S2)
  S3 运行时可变示例库           HITL 置信门控 + Golden-SQL 飞轮
                               Memory 模式捕获 + 衰减排序

依赖:
  S1 ─┬─→ Error 指标   ┌─→ HITL 门控 ──→ Golden-SQL 捕获
      │               │
  S2 ─┼───────────────┼─→ Memory promotion 闸门
      │               └─→ Eval LLM-Judge
  S3 ─┴─→ Golden-SQL 飞轮 / 模式记忆

并行度:阶段0 内 S1‖S2‖S3 可并行;阶段1 内 Error‖HITL‖Memory 可并行;阶段2 依赖 S2。
```

---

## 4. 阶段 0 — 三块共享地基

### S1 可观测性底座 — `openchatbi/observability/`(新包)

**目的**:把"零 tracing / 零 metrics / 零结构化审计"提升到生产可用,且为 Error/HITL/Eval 提供统一的指标与上下文 plumbing。

**模块组成**:

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `context.py` | 运行上下文传播 | `current_user_id`/`current_request_id: ContextVar`;`set_run_context(user_id, request_id)`/`get_run_context()` |
| `metrics.py` | LLM 调用指标 | `@dataclass LLMCallRecord(model, input_tokens, output_tokens, total_tokens, cost_usd, latency_s, node, layer, status)`;`record_llm_call(rec)`;可选 `start_prometheus_exporter(port)` |
| `pricing.py` | 成本估算 | `estimate_cost(model, in_tok, out_tok) -> float`,小型价格表 + config 覆盖 + 未知模型 `cost=0` 兜底 |
| `audit.py` | 结构化审计 | `AuditLogger.log_sql_exec(sql_masked, dialect, row_count, duration_ms, status, user_id, error)`、`log_tool_call(tool, args_masked, result_preview, duration_ms, status, user_id)`;`mask_sql()`/`mask_args()` |
| `callbacks.py` | 工具审计 callback | `class ToolAuditCallback(BaseCallbackHandler)`:`on_tool_start/end/error` → `AuditLogger.log_tool_call` |
| `tracing.py` | tracing 接入 | `get_tracing_callbacks() -> list[BaseCallbackHandler]`、`build_run_config(user_id, session_id, request_id=None, base=None) -> dict` |
| `logging_setup.py` | 日志引导 | `setup_logging(level='INFO', json=True)` 配 stdlib root logger(含 ctx 字段),opt-in 不覆盖既有 handler |
| `streaming.py`(改) | 每轮 token/cost 流式展示 | 在 `StreamEvent` union 增 `StreamUsage{turn_tokens, turn_cost_usd, by_model}`(或复用 `StreamStep(kind='usage')`);`AgentStreamProcessor` 增 `turn_usage` 累加器,从 `_process_message`(L178-202)末块聚合 `usage_metadata`,turn 末 emit |

**tracing provider:Langfuse v3(自托管)**(已核实开发环境自托管实例可达,版本 `3.177.1`;实例地址走 `LANGFUSE_HOST` 环境变量,内网地址不入库):

- 集成:`from langfuse.langchain import CallbackHandler`(**v3 路径**,非 v2 的 `langfuse.callback`);`handler = CallbackHandler()` 从环境变量读取凭据。
- 环境变量(放 `.env`,已 gitignore,**不入库**):`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`(= 自托管 base URL,注意 v3 变量名是 `LANGFUSE_HOST` 而非 `LANGFUSE_BASE_URL`)。
- `config_loader` 当前不加载 dotenv → 在 app 启动点(`run_cli.py` / API)增加一次 `load_dotenv()`。
- LangSmith 作为可选 fallback(`provider: 'langfuse'|'langsmith'|None`),`langsmith` 从 test-only extra 提升为 runtime optional extra `observability`。

**关键接线点**(4 处 invoke config 注入 + LLM/SQL/工具埋点 + 流式消耗展示):

1. `call_llm_chat_model_with_retry`(`openchatbi/llm/llm.py:82-155`,注意是 `llm/llm.py` 子包):成功后读 `response.usage_metadata` + `response_metadata.model_name`,计算成本,`record_llm_call`。新增 `metadata: dict|None` 参数携带 `node_name/layer`。
2. `execute_sql_node`(`generate_sql.py:331-416`):围绕 `_execute_sql` 调用包 `AuditLogger.log_sql_exec`(SQL 脱敏、只记 `row_count` 不记结果体);`user_id` 由 `observability/context.get_run_context()` 读取(**非透传签名**,故 `set_run_context()` 必须在 CLI/API turn 起点调用,不只在 `build_run_config` 内)。
3. **工具审计走 `ToolAuditCallback`**(BaseCallbackHandler),不用装饰器——`run_python_code` 无 `config` 参数,装饰器无法拿到 user 归属;callback 还能零改动覆盖 text2sql/data_analysis/search_knowledge/save_report/MCP/子 agent。
4. `build_run_config()` 注入 **4 个调用点**:`run_cli.py:277-278`(config dict 构造行)、`sample_api/async_api.py:123`、`sample_ui/streaming_ui.py:86/106`、`sample_ui/streamlit_ui.py:71/94`。
5. `config_loader.Config` 增 `observability` 子模型 `{tracing:{enabled,provider,langfuse_host,sample_rate}, metrics:{enabled,prometheus_port}, audit:{enabled,sink,path,mask_sql_literals}}`。
6. `streaming.py` + `run_cli.py`/`async_api.py`:把每轮 token/cost 经 `StreamUsage` 透到 UI 层,renderer 打印 `Turn: N tokens (~$X)`——让运营者不开 dashboard 也能看见消耗(报告明确要求的 operator-facing readout)。

**【核查必改】**:

- ❌ **不要改 `sample_ui/async_graph_manager.py`** —— 它只 build graph,无 invoke/config,是死靶。
- ⚠️ `generate_sql_node`/`regenerate_sql_node` 直接 `llm.invoke()`(`generate_sql.py:315/448`)**绕过 wrapper** → 这两处 token 用 **callback 计数(OpenAICallbackHandler / Langfuse handler)**,不要强行塞进 tool-validation 为主的 wrapper。
- ⚠️ text2sql 子图 trace 连续性靠 **contextvar 隐式传播**(`get_sql_tools` 未给 `sql_graph.invoke/ainvoke` 透传 config,`agent_graph.py:158/175`)→ **二选一**:给 text2sql tool 增 `config` 参数并透传(对齐 `analysis/agent.py` 子 agent 的做法),或加测试显式验证 contextvar 在 sync ToolNode / `asyncio.to_thread` 边界不断。
- ⚠️ `usage_metadata` 各 provider 不一致(streaming / Ollama / Anthropic vs OpenAI)→ 加 `tiktoken` fallback + None 容错。
- ⚠️ `setup_logging` opt-in、不 clobber 既有 handler(避免影响嵌入式使用者)。

**新依赖**(全部进 optional extra `observability`,不进 core):`langfuse>=3`、`langsmith`(提升)、`prometheus-client`(可选)、`tiktoken`(可选 fallback)。

**工作量**:M。

---

### S2 SQL 质量评估器 — `openchatbi/text2sql/confidence.py`(新)

**目的**:一个评估器,三处复用——HITL 置信门控、Eval 的 LLM-as-Judge、Memory promotion 闸门。

**接口**:

```python
@dataclass
class ConfidenceResult:
    score: float          # 0-1
    reasons: list[str]
    checks: dict[str, bool]   # select_columns/where/calc/subquery/joins/exec_intent

class SimpleSQLEvaluator:
    def __init__(self, llm: BaseChatModel | None = None): ...   # 默认 get_default_llm()
    def evaluate(self, question: str, sql: str,
                 schema_info: dict, data_sample: str | None) -> ConfidenceResult: ...
```

- 单次**结构化输出** LLM 调用,跑 Dataherald 的 6 步 rubric(SELECT 列对应 / WHERE 正确 / 计算 / 子查询分解 / JOIN 列匹配 / 执行结果合理)。
- 低温度、结构化输出,保证可复现性尽量高。
- rubric prompt 模板独立(便于将来加重型 `EvaluationAgent` 模式,由 `confidence_evaluator_mode` config 切换)。

**复用现成**:`openchatbi.llm.get_default_llm`,无新 LLM 依赖。

**【核查必改】**:LLM 置信分本身无 ground-truth 校准 → 阈值需可调 + 可整体关闭,避免过度打扰或永不门控。

**工作量**:S(纯新模块,无图改动)。

---

### S3 运行时可变"已学习 SQL 知识库" — 改造 `text2sql_utils.py` + `data.py` + `utils.py`

**目的**:把 `data.py:16` 的静态单例 retriever 变成运行时可写、带来源/打分元数据的统一 store,**同时承载** HITL 的人工批准 golden SQL(`source='golden'`)与 Memory 的自动捕获 SQL 模式(`source='auto'`)。

> **设计决策(统一,偏离初版双 store 草图)**:对抗核查指出初版把 Memory 模式记忆建在独立的 `memory.db`(langgraph BaseStore)上,与 S3 的向量 retriever 是两套 store——这会让"S3 复用两遍"的论证落空,也违背"build once"原则。本方案**统一**为:golden 与 auto 模式都进 S3 这一个向量 store,用 `metadata.source/importance/last_used/use_count` 区分与打分;一次写入路径、一次混合检索、一套打分(`memory_scoring.py`)。Memory 维度对**用户长期记忆(langmem/`memory.db`)**的衰减重排是另一回事,仍留在 `memory.db`(见 §5.3)。

**改造**:

```python
# text2sql_utils.py: 暴露底层 handle
def _init_sql_example_retriever(catalog, vector_db_path) -> tuple[Retriever, dict, VectorStore]:
    # 原返回 (retriever, dict) → 增加 vector_db

# 新增 LearnedSQLStore（统一 golden + auto;HITL/Memory 共用）
class LearnedSQLStore:
    def __init__(self, vector_db, example_dict, lock): ...
    def add(self, question: str, sql: str, tables: list[str], *,
            source: str,                 # 'golden'(人工批准) | 'auto'(S2 闸门捕获)
            importance: float = 1.0,
            namespace: str = 'global') -> None: ...
            # add_texts(metadatas=[{sql,tables,source,importance,use_count,last_used}]) + 原地更新 dict(加锁)
    def retrieve(self, question: str, k: int = 10,
                 score_fn=None) -> list[tuple[str, str, list[str]]]: ...   # score_fn 注入 composite_score
    # 兼容别名:add_golden_sql(q,sql,tables) = add(..., source='golden', importance=high)
```

**【核查必改】**:

- ✅ **双写强制**:运行时 `vector_db.add_texts()` + 持久化 YAML `append`(见 §5.2 HITL 的 `append_sql_example`)。原因:`create_vector_db` 的 cache-invalidation(`utils.py:191-219`)在重启时按文本 count/content 比对重建 collection,只在内存里 add 而没持久化的条目会被丢弃。
- ✅ BM25 路径(`SimpleStore`,无 embedding 时的 fallback)**已确认**支持 `add_texts`(`utils.py:416`,重建 BM25 在 457)+ `max_marginal_relevance_search`(532)→ Chroma / BM25 两路都通。
- ⚠️ `add_texts` 重建 BM25 是 **O(N) 且非线程安全**(原地改 `self.texts/self.bm25`)→ **加锁**,尤其 async server 路径。
- ⚠️ 全局 `text2sql` collection 混所有用户的 golden SQL → 需 `namespace`,默认 `global` 只放 schema 级模式、绝不放 PII。

**工作量**:M。

---

## 5. 阶段 1 — 维度功能接到地基上

### 5.1 Error Handling(3→4)

**新建** `openchatbi/text2sql/errors.py`:

```python
class RecoveryStrategy(str, Enum):
    RETRY = 'retry'; RETRY_WITH_NEW_TABLE = 'retry_with_new_table'
    SURFACE_TO_USER = 'surface_to_user'; ABORT = 'abort'

class Text2SQLError(Exception):
    code: str                        # 取值 = 现有 SQL_* 常量,保持下游兼容
    recovery_strategy: RecoveryStrategy
    user_message: str | None
    orig: BaseException | None

# 子类:SQLSecurityError(沿用,迁入并 re-export)/SQLSyntaxError/InvalidDBConnectionError/
#       DBTimeoutError/EmptyResultError/UnknownSQLError —— 各挂 recovery_strategy
def classify_sql_exception(exc: BaseException) -> Text2SQLError: ...   # 复用现有 _extract_exception_message/_classify_operational_error
```

**改造**:

- `execute_sql_node`:多分支 except 收敛为单点 `classify_sql_exception`;`previous_sql_errors` 条目增 `error_code/recovery_strategy/attempt`。
- `_should_generate_visualization_or_retry`(`sql_graph.py:40-58`):按 `recovery_strategy` 分流;`sql_max_retries` 从 `config` 读(解 `:51` 硬编码)。
- `Config` 增 `sql_max_retries:int=3`、`retry_on_timeout:bool=False`、`retry_strategy_overrides:dict={}`。
- `regenerate_sql_node`:增量增强——每个 Attempt 带 `error_type` + 按类型修正提示(累积上下文本已存在)。

**【核查必改】**:

- ❌ `error_type` 字段被 **7+ 处测试**精确字符串耦合(`'SQL syntax error'`/`'SQL security error'`/`'Database operational error'`)→ **保留原人类可读串**,类名/code 放**新字段**(`error_class`/`error_code`)。
- ⚠️ `RETRY_WITH_NEW_TABLE` 会绕过累积错误 prompt(`table_selection` 出边硬接 `generate_sql` 而非 `regenerate_sql`,`sql_graph.py:106`)→ **列为第二阶段、config 默认关**,实现时需为换表分支单独携带错误上下文。
- ⚠️ `EmptyResultError` 把空结果从 `SQL_SUCCESS` 改软失败,会改变 `generate_visualization_node` 入口 → **默认仍 SUCCESS**,仅开关开启时生效。
- ⚠️ 若主图要消费 `recovery_strategy`(如 `SURFACE_TO_USER`→AskHuman),需同步扩展 `SQLOutputState`(子图输出 schema),否则字段在边界被裁掉。
- ⚠️ `constants.py` 的 `SQL_EXECUTE_TIMEOUT` 常量**值**实际是 `'SQL_CHECK_TIMEOUT'`(名值不一致)→ 用常量符号不要用字面量。

**工作量**:M。

### 5.2 Human-in-the-Loop(4→5)

**置信门控**:

- 新增 `score_sql_node`(调 **S2** `SimpleSQLEvaluator`)+ `confidence_gate` 节点:
  ```python
  # 默认:执行后审查(execute_sql success → score → gate)
  if enable_confidence_gate and sql_confidence < threshold:
      decision = interrupt({'text': f'Low-confidence SQL ({score:.2f}). Approve?',
                            'buttons': ['approve','reject','edit'], 'sql': sql})
  # route_after_confidence: approve→visualization / reject→regenerate_sql / edit→用户 SQL
  ```
- **执行前门控开关**(决策):`config.confidence_gate_mode: 'post_exec'(默认)|'pre_exec'`。`pre_exec` 时图顺序为 `generate_sql→score→gate→execute_sql`(改图结构,标第二阶段)。
- 复用现成 `ask_human` interrupt 通路(`sql_graph.py:23-37`)+ `StreamInterrupt`/`Command(resume)`。

**Golden-SQL 飞轮**:

- `confidence_gate` 的 `approve` 分支调 **S3** `LearnedSQLStore.add(..., source='golden')`(= `add_golden_sql` 别名)+ `catalog.append_sql_example`(durable 双写)。
- `catalog_store.py` 增抽象 `append_sql_example(question, sql, tables, source='golden')`(append + dedup,**不是** `save_table_sql_examples` 的覆盖)。
- `search_knowledge`(`search_knowledge.py:26`)增 `sql_examples` KB 分支,查 S3 `LearnedSQLStore.retrieve` 返回 top-k 已验证 Q→SQL。

**Config / State**:`enable_confidence_gate=False`、`sql_confidence_threshold=0.7`、`enable_golden_sql=False`、`confidence_gate_mode='post_exec'`、`golden_sql_namespace='global'`;`SQLGraphState` + `SQLOutputState` 增 `sql_confidence`、`confidence_reasons`、`human_sql_decision`。

**【核查必改】**:

- ⚠️ `_init_sql_example_retriever` 当前只返回 `(retriever, dict)` → 必须改为暴露 `vector_db` handle(见 S3),否则运行时 append 无落点。
- ⚠️ 双写强制(同 S3);`add_texts` 加锁(同 S3)。
- ⚠️ `search_knowledge` 函数体当前只实现 `columns` 分支(`business` 仅在 docstring,从未分支)→ 加 `sql_examples` 分支时一并修正。
- ⚠️ confidence_gate interrupt 在子图内 fire,经 `get_sql_tools` 的 StructuredTool 捕获 GraphInterrupt 再 raise(`agent_graph.py:160-162`)→ 复用已验证的 AskHuman 传播路径,但需测试穿过 tool 边界后 resume 回正确节点。

**工作量**:M。

### 5.3 Memory & State(4→5)

Memory 维度有两条独立线:**(A) SQL 模式记忆**——并入 S3(`LearnedSQLStore`,`source='auto'`),不另起 store;**(B) 用户长期记忆衰减**——对现有 langmem(`memory.db`)做重要性/衰减重排。共享打分逻辑 `memory_scoring.py`。

**新建** `openchatbi/memory_scoring.py` + `openchatbi/memory_config.py`(均置于包根,与现有 `context_config.py` 同级,**不再有独立 `pattern_memory.py`**——捕获逻辑直接在 `execute_sql_node` 写入 S3):

```python
# memory_scoring.py —— 同时服务 S3 SQL 模式检索 与 langmem 长期记忆重排
def decay_factor(last_used_iso, half_life_days, now=None) -> float   # exp(-ln2*age/half_life)
def composite_score(similarity, importance, last_used_iso, use_count, cfg) -> float
def bump_on_access(meta: dict) -> dict        # use_count+=1; last_used=now
def prune_stale(store, namespace, cfg) -> int

# memory_config.py —— 镜像 context_config.py 的加载方式
@dataclass class MemoryConfig:
    enable_pattern_memory: bool = False        # 默认关(遵守 §2 零回归原则)
    enable_memory_decay_rerank: bool = False   # 默认关
    pattern_scope: str = 'global'
    importance_decay_half_life_days: float = 30.0
    min_retrieval_score: float = 0.2
    max_patterns_per_query: int = 5
    prune_interval_hours: int = 24
def get_memory_config() -> MemoryConfig: ...   # 同 get_context_config()
```

**改造**:

- **(A)** `execute_sql_node` 成功路径:**异步 fire-and-forget** 把成功的 (rewrite_question → SQL → tables) 经 **S3** `LearnedSQLStore.add(source='auto', importance=f(retry_count))` 写入(try/except + flag,绝不阻塞响应)。
- **(A)** `_get_relevant_sql_examples_prompt`:检索改用 S3 `LearnedSQLStore.retrieve(score_fn=composite_score)`——静态 + golden + auto 混合,按 `相似度×重要性×衰减` 排序、按 question 去重、cap `max_patterns_per_query`。
- **(B)** `search_memory_tool`(`tool/memory.py`)包一层按 `composite_score` 重排;`manage_memory` 写入时 stamp `importance/last_used/use_count`。

**【核查必改】**:

- 🔴 **最重要**:"执行成功 ≠ 逻辑正确",自动捕获会毒化 few-shot 池 → **`source='auto'` 写入必须过 S2 置信闸门**(这是 S2 第三处复用的由来)。**依赖顺序:S2 必须先于 Memory 的 auto 捕获启用**;`enable_pattern_memory` 默认 `False`,且仅当 S2 闸门已接线时才允许开启(见 §2 原则 #2、§8 ADR #5)。
- ⚠️ `create_sql_nodes`(`generate_sql.py:136`)签名当前**不接收** S3 store handle(只 `build_sql_graph`/`sql_graph.py:62` 有 `memory_store`)→ 需扩展 `create_sql_nodes` 签名一跳透传,或从 config 读。
- ⚠️ `MemoryConfig`/`observability` 必须先在 `Config` 上**声明字段**:`Config` 是 **pydantic `BaseModel`**(`config_loader.py:26`,非 dataclass),默认 `extra='ignore'`,未声明的子配置传入 `cls(**config)` 会被**静默丢弃**(不报错也读不到)——对齐 `context_config` 在 `config_loader.py:83` 的声明方式。
- ⚠️ `_get_relevant_sql_examples_prompt` 现按 used_tables ⊆ 选中表过滤(`generate_sql.py:205`)→ 混合学到的模式时要兼容或刻意放宽此过滤。
- ⚠️ 无 embedding 时 S3(BM25 fallback 仍可)与 langmem(返回 None,静默禁用)行为不同,需文档化。

**工作量**:M。

---

## 6. 阶段 2 — 证明它:Eval(3→4)

**新建** `evals/judge/`:

```python
# llm_judge.py —— 直接包 S2 的 SimpleSQLEvaluator
class LLMAsJudgeEvaluator:
    def judge(self, question, generated_sql, expected_sql=None, schema=None) -> JudgeVerdict
# run_judge.py —— python -m evals.judge.run_judge --cases ... --out judge_out/report.json
#   按 category 聚合 pass-rate + mean score,跑在 RunLedger 之外、非阻塞 CI(nightly)
```

**扩 RunLedger 语料 1→20+**:

- `cases/*.yaml` 增 `category`(aggregation/join/timerange/anomaly/visualization/text2sql/report)+ `gold.{expected_sql, expected_tool_trajectory, expected_result_contains}`;`cases_path` 自动发现(无需改 suite)。
- `agent.py:_stub_llm_call` 改为**按 prompt 选**的脚本化轨迹驱动(**core:case-id 不在 JSONL 协议里**,只能 key on prompt),让每个 case 跑对应 tool 序列。
- `suite.yaml` `tool_registry` 扩到 5 工具;`tools.py` `TOOLS` 注册 `show_schema/text2sql/run_python_code/save_report` 真实 callable(record 模式)。
- `pyproject.toml` 增 optional extra `eval = ["runledger", "pyyaml"]`(runledger 当前未声明、仅 CI pip install)。
- `.github/workflows/runledger.yml` 增一个**非阻塞 judge job**(`schedule: nightly` + `workflow_dispatch`,带 LLM key secret,上传 `judge_out/report.json`),与现有确定性 replay gate 分离。
- README 补 case/cassette/protocol 格式文档。

**【核查必改】**:

- 🔴 RunLedger **完全 stub LLM** → 扩用例只测管道,**judge 必须与语料同时落地**,否则假覆盖。
- ⚠️ 自定义 assertion 类型 `llm_judge` 在 RunLedger 0.1.1 **无证据支持** → judge 跑在 RunLedger **之外**(更安全)。
- ⚠️ 可复用种子极薄:RunLedger 现役语料 = `cases/t1.yaml` **1 条**;`conftest.py:214-223` 的 2 条 COUNT/AVG 是单测 fixture、非 RunLedger 输入 → gold SQL 基本**手写**,20+ cassette 手录,工作量 **L**。
- ⚠️ RunLedger CI 当前 label-gated(PR 打 `runledger` 标签 或 workflow_dispatch 才跑)→ 扩的语料平时 PR 不跑,需评估是否放开 gate。
- ⚠️ 现有 pytest 测试约 **400 个**(`grep 'def test_'` ≈ 398–404,口径依统计方式;报告的 ~409 略偏高,不影响结论)。

**工作量**:L。

---

## 7. 评分影响与工作量汇总

| 维度 | 现分 | 目标 | 工作量 | 主要借鉴 | 依赖地基 |
|---|---|---|---|---|---|
| Observability | 1 | 3 | M | DB-GPT / MindsDB / Vanna | — (即 S1) |
| Error Handling | 3 | 4 | M | Dataherald / MindsDB | S1 |
| HITL | 4 | 5 | M | Dataherald | S2 + S3 |
| Memory | 4 | 5 | M | Vanna / DB-GPT | S2 + S3 |
| Eval | 3 | 4 | **L** | Vanna / Dataherald | S2 |
| **5 维子分** | **15** | **21** | — | — | (+6) |
| 未改维度(Tool Orch 4 / Security 3) | 7 | 7 | — | 不在本方案范围 | — |
| **Harness 总分** | **22/35** | **28/35** | — | — | — |

---

## 8. 关键决策记录(ADR)

1. **组织方式 = 地基优先分层**(非 P0→P1→P2 顺序)。理由:**S2 被 3 个维度复用**(HITL 门控 / Eval judge / Memory 闸门),**S3 被 2 个维度复用**(HITL golden / Memory auto 模式),顺序执行会把它们各实现两遍。
2. **HITL 门控时机 = 执行后审查为默认 + 执行前门控作为 config 开关**(`confidence_gate_mode`)。理由:`_validate_sql_safety` 强制只读 SELECT,执行后审查改动小、风险低;执行前门控需重排图结构,留作可选第二阶段。
3. **tracing provider = Langfuse v3(自托管)**,LangSmith 可选 fallback。理由:用户提供了自托管实例(已核实可达 `v3.177.1`),内网可用、数据自控。
4. **密钥处理 = 仅环境变量 / `.env`(已 gitignore),不入库**。文档与 `config.yaml.template` 只出现变量名。
5. **所有新特性默认关**,经 config 灰度开启,保证零回归。其中 Memory 的 auto 模式捕获额外要求 **S2 闸门先接线**才允许开启(读路径无回归,写路径靠 S2 防污染)。
6. **S3 统一为单一"已学习 SQL 知识库"**:golden(人工批准)与 auto(S2 闸门捕获)进同一向量 store,用 `metadata.source` 区分;偏离初版"双 store"草图,以兑现 build-once(详见 §4 S3)。Memory 对用户长期记忆(langmem/`memory.db`)的衰减重排是另一条线,仍留在 `memory.db`。

---

## 9. 风险登记

| 风险 | 维度 | 缓解 |
|---|---|---|
| 改 `execute_sql_node` except 链破坏 7+ 测试 | Error | 保留 `error_type` 人类可读串,新信息进新字段;先补测试再改 |
| 自动捕获"成功但逻辑错"的 SQL 毒化示例池 | Memory | promotion 过 S2 置信闸门;provenance 标记 + 可删 |
| 置信评估器每查询加 LLM 调用,增延迟/成本 | HITL | 默认关、可异步并行、可调阈值 |
| 运行时 retriever 写入并发不安全 | HITL/Memory/S3 | 加锁;双写保持 Chroma/YAML 一致 |
| contextvar 不跨线程池/`to_thread` 传播 | S1 | sync(ToolNode)与 async 两路都测;trace 透传二选一 |
| `usage_metadata` 各 provider 不一致 | S1 | tiktoken fallback + None 容错 |
| RunLedger stub LLM 造成假覆盖 | Eval | judge 与语料同时落地,gold SQL 手写锚定 |
| 多租户 golden/pattern 数据串泄 | HITL/Memory | namespace;global 仅放 schema 级、不放 PII |
| Langfuse callback 增网络开销 | S1 | config-gated + sample_rate;默认 dev 才开 |

---

## 10. 研究方法与局限

- **方法**:基于 OpenChatBI worktree 真实源码的代码级深挖(workflow 5 维度并行)+ 对抗式核查(每维度独立 verifier 复核 file:line 论断与设计可行性)。Langfuse 实例已实测可达。
- **局限**:
  - RunLedger(0.1.1,外部包)本地未安装,其 `cases_path` 自动发现、自定义 assertion 等行为为合理推断,实现时需对照安装版本验证。
  - 评分目标(22→28,/35)为设计预期,需落地后实测。
  - 工作量估算(S/M/L)为相对量级,非精确人日。

## 11. 交付物

- 本设计文档:`docs/superpowers/specs/2026-06-09-openchatbi-harness-improvement-design.md`
- 后续:由 writing-plans 产出分 Task 的可执行 implementation plan。
- 落地代码改动范围:
  - **新增**:`openchatbi/observability/`(包:context/metrics/pricing/audit/callbacks/tracing/logging_setup)、`openchatbi/text2sql/confidence.py`(S2)、`openchatbi/text2sql/errors.py`、`openchatbi/memory_scoring.py`、`openchatbi/memory_config.py`、`evals/judge/`(llm_judge/run_judge/rubric)。
  - **改造**(约 13 个):`llm/llm.py`、`streaming.py`、`text2sql/{text2sql_utils,data,generate_sql,sql_graph}.py`、`utils.py`、`tool/{search_knowledge,memory}.py`、`catalog/catalog_store.py`+`catalog/store/file_system.py`、`config_loader.py`、`graph_state.py`、`agent_graph.py`、`run_cli.py`、`sample_api/async_api.py`、`sample_ui/{streaming_ui,streamlit_ui}.py`、`evals/runledger/{suite.yaml,tools.py,agent/agent.py,README.md}`、`.github/workflows/runledger.yml`、`pyproject.toml`、`config.yaml.template`。
  - **明确不动**:`sample_ui/async_graph_manager.py`(build-only 死靶)。
