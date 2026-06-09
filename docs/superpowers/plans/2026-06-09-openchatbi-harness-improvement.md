# OpenChatBI Harness 整体改进方案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把《OpenChatBI Harness 整体改进方案》设计(`docs/superpowers/specs/2026-06-09-openchatbi-harness-improvement-design.md`)落地为 17 个可执行 TDD 任务,将 Harness 总分从 22/35 提升到 28/35。

**Architecture:** 三段式地基优先。阶段 0 先建三块共享地基(S1 可观测性底座 `observability/`、S2 SQL 质量评估器 `text2sql/confidence.py`、S3 统一"已学习 SQL 知识库" `LearnedSQLStore`);阶段 1 把维度功能(Error 结构化错误分类、HITL 置信门控+Golden-SQL 飞轮、Memory 模式记忆+衰减)接到地基上;阶段 2 用 Eval(RunLedger 语料扩充 + LLM-as-Judge)证明。所有新特性默认关,经 config 灰度,保证零回归。

**Tech Stack:** Python · LangGraph · pytest(`uv run pytest`)· Langfuse v3(self-hosted, env 注入)· Chroma/BM25(langchain-chroma + SimpleStore fallback)· langmem · pydantic v2 · deepagents。

---

## File Map

**新增**
- `openchatbi/observability/` — context.py(run-context contextvars)/ logging_setup.py / pricing.py / metrics.py / audit.py / callbacks.py / tracing.py
- `openchatbi/text2sql/confidence.py` — S2 SimpleSQLEvaluator
- `openchatbi/text2sql/errors.py` — RecoveryStrategy + Text2SQLError 层级 + classify_sql_exception
- `openchatbi/memory_scoring.py` / `openchatbi/memory_config.py` — 重要性/衰减打分 + MemoryConfig
- `evals/judge/` — llm_judge.py(LLMAsJudgeEvaluator)/ run_judge.py / rubric.py

**改造**
- `openchatbi/llm/llm.py` — token/cost 埋点;`openchatbi/streaming.py` — StreamUsage 每轮消耗
- `openchatbi/text2sql/{text2sql_utils,data,generate_sql,sql_graph}.py` — S3 retriever 暴露 + LearnedSQLStore + 错误分类 + 门控 + 捕获 + 混合检索
- `openchatbi/utils.py`(create_vector_db 写入)、`openchatbi/tool/{search_knowledge,memory}.py`、`openchatbi/catalog/{catalog_store.py,store/file_system.py}`(append_sql_example)
- `openchatbi/config_loader.py`(pydantic Config 新字段)、`openchatbi/graph_state.py`(SQLGraphState+SQLOutputState)、`openchatbi/agent_graph.py`
- `run_cli.py`、`sample_api/async_api.py`、`sample_ui/{streaming_ui,streamlit_ui}.py`(build_run_config 注入;**不动** async_graph_manager.py)
- `evals/runledger/{suite.yaml,tools.py,agent/agent.py,README.md}`、`.github/workflows/runledger.yml`、`pyproject.toml`、`openchatbi/config.yaml.template`

**依赖顺序**:阶段0(S1‖S2‖S3 可并行)→ 阶段1(Error‖HITL‖Memory;HITL/Memory 依赖 S2+S3)→ 阶段2(Eval 依赖 S2)。Memory 的 auto 捕获(Task 14)必须在 S2 门控接好后才启用。

---

## 实现约定与勘误(Cross-Task Conventions — 执行前必读)

> 本计划由多份草稿合并而成,经对抗式 critic 复核。以下约定**权威优先于任何单个 Task 内的措辞/代码片段**;当某 Task 的片段与此处冲突时,以此处为准。逐 Task 执行时先读本节。

1. **行号是原始文件状态的快照,会随前序 Task 插入代码而漂移。** 定位编辑点请按**命名锚点**(函数名 / `return` 语句 / 具体字符串),不要信任硬编码行号。每个 Task 编辑 `generate_sql.py` 前,先 `grep` 确认目标符号当前位置。

2. **`create_sql_nodes` 的返回 arity:Task 11 起为 6-tuple** `(generate_sql_node, execute_sql_node, regenerate_sql_node, generate_visualization_node, score_sql_node, confidence_gate_node)`。Task 14 对它的改动**只是追加** `learned_sql_store=...` 入参,**不改 arity、不改 `build_sql_graph` 里的 6-名解包、不动 score_sql/confidence_gate 节点注册与 `route_after_confidence` 边**。Task 14 中任何"4-tuple 解包 / `len(nodes)==4`"一律按 6-tuple 修正(`gen, execute_sql_node, regen, viz, score_sql, gate = create_sql_nodes(...)`)。

3. **SQL 执行器接缝 = 闭包 `_execute_sql`(在 `create_sql_nodes` 内,generate_sql.py:246),不要重命名、不要提到模块级。** Task 3 的审计包裹、Task 9 的调用、Task 14 的捕获都在 `execute_sql_node` 内围绕这个闭包做。**测试一律通过 `mock_catalog.get_sql_engine.return_value.connect...` 驱动**(见现有 `tests/test_text2sql_generate_sql.py` 错误路径用例),**禁止**发明 `_run_execute_sql` / `_resolve_execute_node` / `_AUDIT_EXECUTE_SQL_HOOK` / `_execute_sql_for_node` 等不存在的 seam。`create_sql_nodes` 真实签名是 `create_sql_nodes(llm, catalog, dialect, visualization_mode='rule')`,返回 tuple(非 dict)。

4. **S3 暴露函数式 accessor:Task 7 的 `data.py` 必须新增 `def get_learned_sql_store(): return learned_sql_store`**(返回模块级单例或 None)。**所有消费方(Task 12 generate_sql/search_knowledge、Task 14)统一 `from openchatbi.text2sql.data import get_learned_sql_store` 并调用之**,不要 `import` 模块级变量(变量在 import 时定值、不可被测试 patch)。

5. **`LearnedSQLStore.retrieve(score_fn=...)` 的 `score_fn` 签名是 `(metadata: dict, base_rank: int) -> float`。** Task 14 **不要**直接传 `composite_score`(它要 5 个位置参数),用适配器包装:
   ```python
   mem_cfg = get_memory_config()
   score_fn = lambda meta, base_rank: composite_score(
       1.0 / (1.0 + base_rank),
       float(meta.get("importance", 1.0) or 1.0),
       meta.get("last_used", ""),
       int(meta.get("use_count", 0) or 0),
       mem_cfg,
   )
   ```
   (Task 15 直接以 5 参调用 `composite_score` 是对的,只有 Task 14 的 score_fn 路径需要适配。)

6. **EmptyResultError 契约**:`EmptyResultError.code = SQL_NA`,`recovery_strategy = RETRY_WITH_NEW_TABLE`。`execute_sql_node` 仅当 `enable_empty_result_gate`(默认 **False**,新增 Config 字段)开启时,对 0 行结果 `raise EmptyResultError` 并返回 `err.code`(=SQL_NA);**默认关闭时 0 行结果仍走 `SQL_SUCCESS`**、正常进可视化。Task 8 对应单元测试写真实断言:`assert EmptyResultError("x").code == SQL_NA` 且 `assert EmptyResultError("x").recovery_strategy is RecoveryStrategy.RETRY_WITH_NEW_TABLE`(删除任何 `if False` / 占位断言)。

7. **超时路由用符号不用字面量**:`constants.py` 的 `SQL_EXECUTE_TIMEOUT` 其值为字符串 `"SQL_CHECK_TIMEOUT"`(名值不一致)。所有比较/赋值一律用导入的符号 `SQL_EXECUTE_TIMEOUT`。Task 10 须含一条用例:构造 `sql_execution_result=SQL_EXECUTE_TIMEOUT`(符号)断言路由到 end(不重试)。

8. **直接 `llm.invoke()` 的 token 计量边界**:`generate_sql_node`/`regenerate_sql_node`(generate_sql.py:315/448)绕过 wrapper,其 token **仅在 tracing(Langfuse/LangSmith)启用时**由 callback 记录;wrapper 覆盖其余调用。`metrics.record_llm_call` 在默认(无 tracing)配置下**不覆盖**这两处——这是已知、可接受的范围限定,不要在文档里宣称默认全覆盖。(可选增强:Task 4 的 `build_run_config` 追加一个轻量 `BaseCallbackHandler` 把 token 喂给 `record_llm_call`,使其不依赖 tracing 开关。)

9. **新测试文件平铺在 `tests/` 下**(与现有布局一致):`tests/test_confidence.py`、`tests/test_learned_sql_store.py` 等,**不要**建 `tests/text2sql/` 子目录;若某 Task 确需子目录(如 `tests/observability/`、`tests/eval/`),必须显式新增该目录的 `__init__.py`。Task 14 Step 9 的回归命令改为 `uv run pytest tests/test_text2sql_generate_sql.py -v`(**不存在** `tests/test_generate_sql.py` / `tests/test_sql_graph.py`)。

10. **HITL 编辑环 × Memory 自动捕获的交互**:当 `enable_confidence_gate` 与 `enable_pattern_memory` **同时开启**时,`route_after_confidence` 的 `edit` 分支会让 SQL 重入 `execute_sql_node`。`_maybe_capture_pattern`(Task 14)**只在终态 success / `human_sql_decision == 'approve'` 时捕获**,对未审批的 edit 重入是 no-op,避免把未批准 SQL 写入示例池(兑现 §5.3 "成功≠正确")。

11. **RunLedger label-gate 决策**:确定性 replay 语料**保持 label-gated**(打 `runledger` 标签或 `workflow_dispatch` 才在 PR 跑),并由 Task 17 的 **nightly** job 兜底全量跑;Task 16 的脚本化驱动须断言 `_TRAJECTORIES` 的 prompt key 无碰撞(`len(_TRAJECTORIES) == 去重 prompt 数`)。

---

## S1-observability(Tasks 1-5)

### Task 1: Observability run-context + JSON logging substrate

**Files:**
- Create: `openchatbi/observability/__init__.py`
- Create: `openchatbi/observability/context.py`
- Create: `openchatbi/observability/logging_setup.py`
- Create: `tests/observability/__init__.py`
- Create: `tests/observability/test_context.py`
- Create: `tests/observability/test_logging_setup.py`
- Modify: `run_cli.py` (wire `set_run_context` at turn start in `run_turn_sync`/`run_turn_async`, ~L218 / ~L231)
- Modify: `sample_api/async_api.py` (wire `set_run_context` at top of `chat_stream`, ~L121)

- [ ] **Step 1: Write failing test for `context.py`.** Create `tests/observability/__init__.py` (empty) and `tests/observability/test_context.py`:
  ```python
  """Tests for observability run-context contextvars."""

  import asyncio

  from openchatbi.observability.context import (
      current_request_id,
      current_user_id,
      get_run_context,
      set_run_context,
  )


  def test_defaults_are_none() -> None:
      # Fresh contextvar reads must not raise and default to None.
      assert current_user_id.get() is None
      assert current_request_id.get() is None
      assert get_run_context() == (None, None)


  def test_set_run_context_roundtrips() -> None:
      set_run_context("alice", "req-123")
      assert get_run_context() == ("alice", "req-123")
      assert current_user_id.get() == "alice"
      assert current_request_id.get() == "req-123"


  def test_set_run_context_isolated_per_task() -> None:
      # Each asyncio task gets its own contextvar copy → no cross-talk.
      async def worker(uid: str) -> tuple[str | None, str | None]:
          set_run_context(uid, f"req-{uid}")
          await asyncio.sleep(0)
          return get_run_context()

      async def main() -> list[tuple[str | None, str | None]]:
          return await asyncio.gather(worker("u1"), worker("u2"))

      results = asyncio.run(main())
      assert ("u1", "req-u1") in results
      assert ("u2", "req-u2") in results
  ```

- [ ] **Step 2: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_context.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability'`.

- [ ] **Step 3: Implement `context.py` and the package `__init__`.** Create `openchatbi/observability/__init__.py`:
  ```python
  """Observability substrate: run-context, metrics, audit, tracing (opt-in)."""
  ```
  Create `openchatbi/observability/context.py`:
  ```python
  """Run-context propagation via contextvars.

  These are populated once at the start of each CLI/API turn so that deep code
  (e.g. ``execute_sql_node``) can attribute work to a user/request *without*
  threading ``user_id`` through every function signature. ContextVars copy into
  asyncio tasks and ``contextvars.copy_context()`` (used by LangGraph's sync
  ToolNode / ``asyncio.to_thread`` boundaries), so trace continuity holds.
  """

  from __future__ import annotations

  from contextvars import ContextVar

  current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
  current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)


  def set_run_context(user_id: str | None, request_id: str | None) -> None:
      """Bind the current user/request ids for the active context."""
      current_user_id.set(user_id)
      current_request_id.set(request_id)


  def get_run_context() -> tuple[str | None, str | None]:
      """Return ``(user_id, request_id)`` for the active context."""
      return current_user_id.get(), current_request_id.get()
  ```

- [ ] **Step 4: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_context.py -v`
  - Expected: PASS — 3 passed.

- [ ] **Step 5: Write failing test for `logging_setup.py`.** Create `tests/observability/test_logging_setup.py`:
  ```python
  """Tests for opt-in JSON logging setup."""

  import json
  import logging

  from openchatbi.observability.context import set_run_context
  from openchatbi.observability.logging_setup import setup_logging


  def test_setup_logging_does_not_clobber_existing_handlers() -> None:
      root = logging.getLogger()
      sentinel = logging.NullHandler()
      root.addHandler(sentinel)
      try:
          setup_logging(level="INFO", json=True)
          # Opt-in setup must keep handlers an embedding host already installed.
          assert sentinel in root.handlers
      finally:
          root.removeHandler(sentinel)
          # Remove the handler our setup added so other tests are unaffected.
          for h in list(root.handlers):
              if getattr(h, "_openchatbi_obs", False):
                  root.removeHandler(h)


  def test_setup_logging_emits_json_with_context_fields(capsys) -> None:
      set_run_context("bob", "req-9")
      setup_logging(level="INFO", json=True)
      logging.getLogger("openchatbi.test").info("hello")
      err = capsys.readouterr().err
      line = [ln for ln in err.splitlines() if "hello" in ln][-1]
      payload = json.loads(line)
      assert payload["message"] == "hello"
      assert payload["level"] == "INFO"
      assert payload["user_id"] == "bob"
      assert payload["request_id"] == "req-9"
      root = logging.getLogger()
      for h in list(root.handlers):
          if getattr(h, "_openchatbi_obs", False):
              root.removeHandler(h)
  ```

- [ ] **Step 6: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_logging_setup.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.logging_setup'`.

- [ ] **Step 7: Implement `logging_setup.py`.** Create `openchatbi/observability/logging_setup.py`:
  ```python
  """Opt-in structured (JSON) logging for the stdlib root logger.

  Intentionally NOT called on import: embedding hosts keep their own logging.
  ``setup_logging`` only adds a handler when the root has none of ours, never
  removes existing handlers, and injects run-context fields into every record.
  """

  from __future__ import annotations

  import json
  import logging
  import sys

  from openchatbi.observability.context import get_run_context


  class _JsonFormatter(logging.Formatter):
      def format(self, record: logging.LogRecord) -> str:
          user_id, request_id = get_run_context()
          payload = {
              "ts": self.formatTime(record),
              "level": record.levelname,
              "logger": record.name,
              "message": record.getMessage(),
              "user_id": user_id,
              "request_id": request_id,
          }
          if record.exc_info:
              payload["exc_info"] = self.formatException(record.exc_info)
          return json.dumps(payload, ensure_ascii=False, default=str)


  def setup_logging(level: str = "INFO", json: bool = True) -> None:
      """Configure the root logger once (opt-in; never clobbers existing handlers)."""
      root = logging.getLogger()
      if any(getattr(h, "_openchatbi_obs", False) for h in root.handlers):
          return
      handler = logging.StreamHandler(stream=sys.stderr)
      handler._openchatbi_obs = True  # type: ignore[attr-defined]
      if json:
          handler.setFormatter(_JsonFormatter())
      else:
          handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
      root.addHandler(handler)
      root.setLevel(getattr(logging, level.upper(), logging.INFO))
  ```

- [ ] **Step 8: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_logging_setup.py -v`
  - Expected: PASS — 2 passed.

- [ ] **Step 9: Wire `set_run_context` at the CLI turn start.** In `run_cli.py`, add the import near the existing `openchatbi.streaming` import block (after L47):
  ```python
  from openchatbi.observability.context import set_run_context
  ```
  Then in `run_turn_sync` (the body begins at L218 with `processor = AgentStreamProcessor()`), insert immediately after that line:
  ```python
      cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
      set_run_context(cfg.get("user_id"), cfg.get("thread_id"))
  ```
  Apply the identical two-line insertion at the top of `run_turn_async` (after its own `processor = AgentStreamProcessor()` at ~L231).

- [ ] **Step 10: Wire `set_run_context` at the API turn start.** In `sample_api/async_api.py`, add to the imports (after the `openchatbi.streaming` block, ~L21):
  ```python
  from openchatbi.observability.context import set_run_context
  ```
  Then inside `chat_stream`, immediately after the `config = {"configurable": ...}` line (L123):
  ```python
      set_run_context(user_id, user_session_id)
  ```

- [ ] **Step 11: Run the full observability + CLI/streaming tests to confirm no regression.**
  - Run: `uv run pytest tests/observability/ tests/test_streaming.py -v`
  - Expected: PASS — all green.

- [ ] **Step 12: Commit.**
  - Run: `git checkout -b s1-observability && git add openchatbi/observability/__init__.py openchatbi/observability/context.py openchatbi/observability/logging_setup.py tests/observability/ run_cli.py sample_api/async_api.py && git commit -m "feat(observability): run-context contextvars + opt-in JSON logging (S1 Task 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

### Task 2: LLM cost pricing + call metrics, wired into the LLM wrapper

**Files:**
- Create: `openchatbi/observability/pricing.py`
- Create: `openchatbi/observability/metrics.py`
- Create: `tests/observability/test_pricing.py`
- Create: `tests/observability/test_metrics.py`
- Modify: `openchatbi/llm/llm.py` (capture usage in `call_llm_chat_model_with_retry`, signature at L82-84; success block at L121-123)

- [ ] **Step 1: Write failing test for `pricing.py`.** Create `tests/observability/test_pricing.py`:
  ```python
  """Tests for cost estimation."""

  from openchatbi.observability.pricing import estimate_cost


  def test_known_model_cost() -> None:
      # gpt-4o: $2.5/1M input, $10/1M output → 1000 in + 500 out.
      cost = estimate_cost("gpt-4o", 1000, 500)
      assert abs(cost - (1000 / 1_000_000 * 2.5 + 500 / 1_000_000 * 10.0)) < 1e-9


  def test_prefix_match_is_case_insensitive() -> None:
      # Provider-prefixed / versioned names resolve via prefix lookup.
      assert estimate_cost("GPT-4o-2024-08-06", 1000, 1000) > 0.0


  def test_unknown_model_returns_zero() -> None:
      assert estimate_cost("some-local-ollama-model", 9999, 9999) == 0.0
  ```

- [ ] **Step 2: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_pricing.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.pricing'`.

- [ ] **Step 3: Implement `pricing.py`.** Create `openchatbi/observability/pricing.py`:
  ```python
  """USD cost estimation for LLM calls.

  A tiny static table (USD per 1M tokens) covering the providers OpenChatBI
  ships with; unknown models fall back to 0.0 so cost accounting never crashes
  on a local/Ollama model. Lookup is case-insensitive longest-prefix so that
  versioned names (``gpt-4o-2024-08-06``) resolve to their family price.
  """

  from __future__ import annotations

  # (input_per_1m, output_per_1m) in USD.
  _PRICES: dict[str, tuple[float, float]] = {
      "gpt-4o": (2.5, 10.0),
      "gpt-4o-mini": (0.15, 0.6),
      "gpt-4.1": (2.0, 8.0),
      "gpt-4.1-mini": (0.4, 1.6),
      "o3": (2.0, 8.0),
      "claude-3-5-sonnet": (3.0, 15.0),
      "claude-3-5-haiku": (0.8, 4.0),
      "claude-sonnet-4": (3.0, 15.0),
      "claude-opus-4": (15.0, 75.0),
  }


  def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
      """Estimate USD cost; unknown models return 0.0."""
      if not model:
          return 0.0
      name = model.lower()
      best: tuple[float, float] | None = None
      best_len = -1
      for prefix, price in _PRICES.items():
          if name.startswith(prefix) and len(prefix) > best_len:
              best, best_len = price, len(prefix)
      if best is None:
          return 0.0
      in_rate, out_rate = best
      return input_tokens / 1_000_000 * in_rate + output_tokens / 1_000_000 * out_rate
  ```

- [ ] **Step 4: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_pricing.py -v`
  - Expected: PASS — 3 passed.

- [ ] **Step 5: Write failing test for `metrics.py`.** Create `tests/observability/test_metrics.py`:
  ```python
  """Tests for LLM call metrics recording."""

  from openchatbi.observability import metrics
  from openchatbi.observability.metrics import LLMCallRecord, record_llm_call


  def test_record_llm_call_appends(monkeypatch) -> None:
      captured: list[LLMCallRecord] = []
      monkeypatch.setattr(metrics, "_RECORDS", captured, raising=False)
      rec = LLMCallRecord(
          model="gpt-4o",
          input_tokens=10,
          output_tokens=5,
          total_tokens=15,
          cost_usd=0.0001,
          latency_s=1.2,
          node="generate_sql",
          layer="text2sql",
          status="success",
      )
      record_llm_call(rec)
      assert captured == [rec]
      assert captured[0].total_tokens == 15
      assert captured[0].status == "success"


  def test_record_llm_call_never_raises() -> None:
      # Recording must be best-effort: a malformed record must not propagate.
      record_llm_call(None)  # type: ignore[arg-type]
  ```

- [ ] **Step 6: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_metrics.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.metrics'`.

- [ ] **Step 7: Implement `metrics.py`.** Create `openchatbi/observability/metrics.py`:
  ```python
  """LLM call metrics: an in-process record sink + optional Prometheus exporter."""

  from __future__ import annotations

  from dataclasses import dataclass

  from openchatbi.utils import log


  @dataclass
  class LLMCallRecord:
      """One LLM invocation's accounting record."""

      model: str
      input_tokens: int
      output_tokens: int
      total_tokens: int
      cost_usd: float
      latency_s: float
      node: str | None
      layer: str | None
      status: str


  # In-process ring of recent records (kept tiny; real sinks are Prometheus/trace).
  _RECORDS: list[LLMCallRecord] = []
  _MAX_RECORDS = 1000


  def record_llm_call(rec: LLMCallRecord) -> None:
      """Record an LLM call (best-effort; never raises into the call path)."""
      try:
          if rec is None:
              return
          _RECORDS.append(rec)
          if len(_RECORDS) > _MAX_RECORDS:
              del _RECORDS[0 : len(_RECORDS) - _MAX_RECORDS]
      except Exception as exc:  # pragma: no cover - defensive
          log(f"record_llm_call failed: {exc!r}")


  def start_prometheus_exporter(port: int) -> None:
      """Start a Prometheus HTTP exporter if prometheus_client is installed (optional)."""
      try:
          from prometheus_client import start_http_server
      except ImportError:
          log("prometheus_client not installed; skipping exporter. Install openchatbi[observability].")
          return
      start_http_server(port)
  ```

- [ ] **Step 8: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_metrics.py -v`
  - Expected: PASS — 2 passed.

- [ ] **Step 9: Write failing test for usage capture in the LLM wrapper.** Create `tests/observability/test_llm_usage_capture.py`:
  ```python
  """Token/cost capture inside call_llm_chat_model_with_retry."""

  from langchain_core.messages import AIMessage

  from openchatbi.llm import llm as llm_mod
  from openchatbi.observability import metrics
  from openchatbi.observability.metrics import LLMCallRecord


  class _UsageModel:
      """Minimal chat model returning a response with usage_metadata."""

      def invoke(self, messages, config=None):
          return AIMessage(
              content="SELECT 1",
              usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
              response_metadata={"model_name": "gpt-4o"},
          )


  def test_wrapper_records_usage(monkeypatch) -> None:
      captured: list[LLMCallRecord] = []
      monkeypatch.setattr(metrics, "record_llm_call", lambda rec: captured.append(rec))
      monkeypatch.setattr(llm_mod, "record_llm_call", metrics.record_llm_call, raising=False)

      resp = llm_mod.call_llm_chat_model_with_retry(
          _UsageModel(), [{"role": "user", "content": "hi"}], metadata={"node_name": "llm_node", "layer": "main"}
      )
      assert resp.content == "SELECT 1"
      assert len(captured) == 1
      rec = captured[0]
      assert rec.model == "gpt-4o"
      assert rec.input_tokens == 12
      assert rec.output_tokens == 4
      assert rec.total_tokens == 16
      assert rec.node == "llm_node"
      assert rec.layer == "main"
      assert rec.status == "success"
      assert rec.cost_usd > 0.0
  ```

- [ ] **Step 10: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_llm_usage_capture.py -v`
  - Expected: FAIL — `TypeError: call_llm_chat_model_with_retry() got an unexpected keyword argument 'metadata'`.

- [ ] **Step 11: Add the `metadata` param and usage capture in `call_llm_chat_model_with_retry`.** In `openchatbi/llm/llm.py`, add imports after L10 (`from openchatbi.utils import log`):
  ```python
  from openchatbi.observability.metrics import LLMCallRecord, record_llm_call
  from openchatbi.observability.pricing import estimate_cost
  ```
  Change the signature at L82-84 to accept `metadata`:
  ```python
  def call_llm_chat_model_with_retry(
      chat_model: BaseChatModel, messages, streaming_tokens=False, bound_tools=None, parallel_tool_call=False,
      metadata: dict | None = None,
  ):
  ```
  Then in the success branch (right after L122-123, `run_time = int(time.time() - start_time)` / the success `log`), insert usage capture (keep the `int(...)` log line; add a precise float for the record):
  ```python
              elapsed = time.time() - start_time
              run_time = int(elapsed)
              log(f"LLM response after {run_time} seconds.")
              try:
                  usage = getattr(response, "usage_metadata", None) or {}
                  in_tok = int(usage.get("input_tokens", 0) or 0)
                  out_tok = int(usage.get("output_tokens", 0) or 0)
                  total_tok = int(usage.get("total_tokens", in_tok + out_tok) or (in_tok + out_tok))
                  model_name = (getattr(response, "response_metadata", None) or {}).get("model_name", "") or ""
                  meta = metadata or {}
                  record_llm_call(
                      LLMCallRecord(
                          model=model_name,
                          input_tokens=in_tok,
                          output_tokens=out_tok,
                          total_tokens=total_tok,
                          cost_usd=estimate_cost(model_name, in_tok, out_tok),
                          latency_s=elapsed,
                          node=meta.get("node_name"),
                          layer=meta.get("layer"),
                          status="success",
                      )
                  )
              except Exception as exc:  # pragma: no cover - never break the call path
                  log(f"LLM usage capture failed: {exc!r}")
  ```
  (Replace the existing two lines `run_time = int(time.time() - start_time)` and `log(f"LLM response after {run_time} seconds.")` at L122-123 with the block above so `elapsed` is available.)

- [ ] **Step 12: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_llm_usage_capture.py -v`
  - Expected: PASS — 1 passed.

- [ ] **Step 13: Note on the generate_sql/regenerate_sql direct-invoke path.** Add a docstring-only comment in `metrics.py` `record_llm_call` referencing that `generate_sql_node`/`regenerate_sql_node` (`generate_sql.py:315/448`) call `llm.invoke()` directly and are covered via the tracing callback counter (Task 4), not this wrapper. Insert after the `_MAX_RECORDS` line:
  ```python
  # NOTE: generate_sql_node / regenerate_sql_node call ``llm.invoke()`` directly
  # (generate_sql.py:315/448), bypassing call_llm_chat_model_with_retry. Their
  # token usage is captured by the tracing callbacks registered in build_run_config
  # (Task 4), NOT by routing them through this wrapper.
  ```

- [ ] **Step 14: Run the full LLM + observability suite to confirm no regression.**
  - Run: `uv run pytest tests/observability/ -v`
  - Expected: PASS — all green.

- [ ] **Step 15: Commit.**
  - Run: `git add openchatbi/observability/pricing.py openchatbi/observability/metrics.py openchatbi/llm/llm.py tests/observability/test_pricing.py tests/observability/test_metrics.py tests/observability/test_llm_usage_capture.py && git commit -m "feat(observability): pricing + LLMCallRecord metrics, wired into call_llm_chat_model_with_retry (S1 Task 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

### Task 3: Structured audit logger + tool-audit callback, wired into SQL execution

**Files:**
- Create: `openchatbi/observability/audit.py`
- Create: `openchatbi/observability/callbacks.py`
- Create: `tests/observability/test_audit.py`
- Create: `tests/observability/test_callbacks.py`
- Modify: `openchatbi/text2sql/generate_sql.py` (wrap `_execute_sql` in `execute_sql_node`, L331-416; user_id via `get_run_context()`)

- [ ] **Step 1: Write failing test for `audit.py`.** Create `tests/observability/test_audit.py`:
  ```python
  """Tests for the structured audit logger + SQL/arg masking."""

  from openchatbi.observability.audit import AuditLogger, mask_args, mask_sql


  def test_mask_sql_redacts_string_and_number_literals() -> None:
      masked = mask_sql("SELECT * FROM users WHERE name = 'alice' AND age = 42")
      assert "alice" not in masked
      assert "42" not in masked
      assert "SELECT" in masked and "FROM users" in masked


  def test_mask_args_redacts_values_keeps_keys() -> None:
      out = mask_args({"question": "who is alice", "token": "secret123"})
      assert set(out.keys()) == {"question", "token"}
      assert out["token"] != "secret123"


  def test_log_sql_exec_emits_structured_record(caplog) -> None:
      import logging

      logger = AuditLogger()
      with caplog.at_level(logging.INFO, logger="openchatbi.audit"):
          logger.log_sql_exec(
              sql="SELECT COUNT(*) FROM users WHERE id = 7",
              dialect="presto",
              row_count=1,
              duration_ms=12.5,
              status="SQL_SUCCESS",
              user_id="alice",
          )
      assert any("SQL_SUCCESS" in r.message for r in caplog.records)
      # Raw literal must never reach the audit sink.
      assert all("= 7" not in r.message for r in caplog.records)
  ```

- [ ] **Step 2: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_audit.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.audit'`.

- [ ] **Step 3: Implement `audit.py`.** Create `openchatbi/observability/audit.py`:
  ```python
  """Structured audit logging for SQL executions and tool calls.

  Never logs result bodies — only ``row_count`` and a short result preview. SQL
  literals and tool-arg values are masked by default so PII / secrets never reach
  the audit sink. Writes JSON lines to the ``openchatbi.audit`` logger.
  """

  from __future__ import annotations

  import json
  import logging
  import re
  from typing import Any

  _audit_logger = logging.getLogger("openchatbi.audit")

  _STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
  _NUMBER_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


  def mask_sql(sql: str) -> str:
      """Redact string and numeric literals from SQL, preserving structure."""
      if not sql:
          return sql
      masked = _STRING_LITERAL.sub("'?'", sql)
      masked = _NUMBER_LITERAL.sub("?", masked)
      return masked


  def mask_args(d: dict) -> dict:
      """Redact values of a tool-arg dict, keeping keys for traceability."""
      out: dict[str, Any] = {}
      for k, v in (d or {}).items():
          out[k] = "<redacted>" if isinstance(v, str) and v else ("<redacted>" if v else v)
      return out


  class AuditLogger:
      """Emits one structured JSON line per audited event."""

      def __init__(self, mask_literals: bool = True) -> None:
          self._mask = mask_literals

      def log_sql_exec(
          self,
          sql: str,
          dialect: str,
          row_count: int | None,
          duration_ms: float,
          status: str,
          user_id: str | None,
          error: str | None = None,
      ) -> None:
          payload = {
              "event": "sql_exec",
              "sql": mask_sql(sql) if self._mask else sql,
              "dialect": dialect,
              "row_count": row_count,
              "duration_ms": round(duration_ms, 2),
              "status": status,
              "user_id": user_id,
              "error": error,
          }
          _audit_logger.info(json.dumps(payload, ensure_ascii=False, default=str))

      def log_tool_call(
          self,
          tool: str,
          args: dict,
          result_preview: str,
          duration_ms: float,
          status: str,
          user_id: str | None,
      ) -> None:
          payload = {
              "event": "tool_call",
              "tool": tool,
              "args": mask_args(args) if self._mask else (args or {}),
              "result_preview": (result_preview or "")[:300],
              "duration_ms": round(duration_ms, 2),
              "status": status,
              "user_id": user_id,
          }
          _audit_logger.info(json.dumps(payload, ensure_ascii=False, default=str))
  ```

- [ ] **Step 4: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_audit.py -v`
  - Expected: PASS — 3 passed.

- [ ] **Step 5: Write failing test for `callbacks.py`.** Create `tests/observability/test_callbacks.py`:
  ```python
  """Tests for the tool-audit callback handler."""

  from uuid import uuid4

  from openchatbi.observability.callbacks import ToolAuditCallback


  def test_callback_logs_tool_call_on_end(monkeypatch) -> None:
      calls: list[dict] = []

      class _Capture:
          def log_tool_call(self, tool, args, result_preview, duration_ms, status, user_id):
              calls.append(
                  {"tool": tool, "preview": result_preview, "status": status, "user_id": user_id}
              )

      cb = ToolAuditCallback(audit=_Capture())
      run_id = uuid4()
      # run_python_code has no config param → callback still attributes it.
      cb.on_tool_start(
          {"name": "run_python_code"}, "print(1)", run_id=run_id,
          inputs={"code": "print(1)"},
      )
      cb.on_tool_end("hello-world-result", run_id=run_id)

      assert len(calls) == 1
      assert calls[0]["tool"] == "run_python_code"
      assert "hello-world-result" in calls[0]["preview"]
      assert calls[0]["status"] == "success"


  def test_callback_logs_error(monkeypatch) -> None:
      calls: list[str] = []

      class _Capture:
          def log_tool_call(self, tool, args, result_preview, duration_ms, status, user_id):
              calls.append(status)

      cb = ToolAuditCallback(audit=_Capture())
      run_id = uuid4()
      cb.on_tool_start({"name": "text2sql"}, "q", run_id=run_id)
      cb.on_tool_error(ValueError("boom"), run_id=run_id)
      assert calls == ["error"]
  ```

- [ ] **Step 6: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_callbacks.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.callbacks'`.

- [ ] **Step 7: Implement `callbacks.py`.** Create `openchatbi/observability/callbacks.py`:
  ```python
  """LangChain callback that audits every tool call.

  Registered once via ``build_run_config`` → ``config['callbacks']`` so it covers
  ALL tools (text2sql / data_analysis / search_knowledge / save_report / MCP /
  sub-agents) including ``run_python_code`` which has no ``config`` param and so
  cannot be covered by a decorator.
  """

  from __future__ import annotations

  import time
  from typing import Any
  from uuid import UUID

  from langchain_core.callbacks import BaseCallbackHandler

  from openchatbi.observability.audit import AuditLogger
  from openchatbi.observability.context import get_run_context


  class ToolAuditCallback(BaseCallbackHandler):
      """Maps on_tool_start/end/error onto ``AuditLogger.log_tool_call``."""

      def __init__(self, audit: AuditLogger | None = None) -> None:
          self._audit = audit or AuditLogger()
          self._pending: dict[UUID, dict[str, Any]] = {}

      def on_tool_start(
          self,
          serialized: dict[str, Any],
          input_str: str,
          *,
          run_id: UUID,
          inputs: dict[str, Any] | None = None,
          **kwargs: Any,
      ) -> None:
          name = (serialized or {}).get("name") or "tool"
          self._pending[run_id] = {
              "name": name,
              "args": inputs or {"input": input_str},
              "start": time.time(),
          }

      def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
          self._finish(run_id, status="success", result_preview=str(output))

      def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
          self._finish(run_id, status="error", result_preview=repr(error))

      def _finish(self, run_id: UUID, status: str, result_preview: str) -> None:
          info = self._pending.pop(run_id, None)
          if info is None:
              return
          user_id, _ = get_run_context()
          duration_ms = (time.time() - info["start"]) * 1000.0
          self._audit.log_tool_call(
              tool=info["name"],
              args=info["args"],
              result_preview=result_preview,
              duration_ms=duration_ms,
              status=status,
              user_id=user_id,
          )
  ```

- [ ] **Step 8: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_callbacks.py -v`
  - Expected: PASS — 2 passed.

- [ ] **Step 9: Write failing test for SQL-exec auditing inside `execute_sql_node`.** Create `tests/observability/test_execute_sql_audit.py`:
  ```python
  """execute_sql_node emits a masked SQL audit record via AuditLogger."""

  import logging

  from openchatbi.observability.context import set_run_context
  from openchatbi.text2sql import generate_sql as gs


  def test_execute_sql_node_audits_success(mock_config, monkeypatch, caplog) -> None:
      set_run_context("alice", "req-1")

      # Build the node closures via create_sql_nodes; stub _execute_sql at module level.
      nodes = gs.create_sql_nodes(mock_config.catalog_store, mock_config.default_llm)
      execute_sql_node = nodes["execute_sql"] if isinstance(nodes, dict) else gs._resolve_execute_node(nodes)

      monkeypatch.setattr(
          gs, "_AUDIT_EXECUTE_SQL_HOOK", None, raising=False
      )
      with caplog.at_level(logging.INFO, logger="openchatbi.audit"):
          state = {"sql": "SELECT COUNT(*) FROM users WHERE id = 7", "previous_sql_errors": []}
          # _execute_sql is closed over in create_sql_nodes; patch the inner runner.
          monkeypatch.setattr(gs, "_run_execute_sql", lambda sql: ({"columns": ["c"]}, "c\n1"), raising=False)
          result = execute_sql_node(state)

      assert result["sql_execution_result"] == "SQL_SUCCESS"
      audit_lines = [r.message for r in caplog.records if r.name == "openchatbi.audit"]
      assert any("sql_exec" in m and "SQL_SUCCESS" in m for m in audit_lines)
      assert all("= 7" not in m for m in audit_lines)
  ```
  Note: `create_sql_nodes` returns node callables; this test asserts the audit side-effect regardless of return container. If `create_sql_nodes` returns a tuple, replace the `nodes[...]` line with the actual unpacking once the signature is confirmed at implementation time (`generate_sql.py:136`).

- [ ] **Step 10: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_execute_sql_audit.py -v`
  - Expected: FAIL — no `openchatbi.audit` record emitted (audit not yet wired).

- [ ] **Step 11: Wire `AuditLogger.log_sql_exec` into `execute_sql_node`.** In `openchatbi/text2sql/generate_sql.py`, add imports after the existing `from openchatbi.utils import ...` line (after L27):
  ```python
  import time as _time

  from openchatbi.observability.audit import AuditLogger
  from openchatbi.observability.context import get_run_context

  _audit_logger = AuditLogger()
  ```
  In `execute_sql_node`, wrap the `_execute_sql` success path (L344-356). Replace the `try:` block opening through the success `return` with timing + audit:
  ```python
          dialect = config.get().dialect
          user_id, _ = get_run_context()
          start = _time.time()
          try:
              schema_info, csv_result = _execute_sql(sql_query)
              duration_ms = (_time.time() - start) * 1000.0
              row_count = schema_info.get("row_count") if isinstance(schema_info, dict) else None
              _audit_logger.log_sql_exec(
                  sql=sql_query, dialect=dialect, row_count=row_count,
                  duration_ms=duration_ms, status=SQL_SUCCESS, user_id=user_id,
              )
              if "result_limit" in schema_info:
                  result_label = f"SQL Result (limited to first {schema_info['result_limit']} rows)"
              else:
                  result_label = "SQL Result"
              result = f"```sql\n{sql_query}\n```\n{result_label}:\n```csv\n{csv_result}\n```"
              return {
                  "sql_execution_result": SQL_SUCCESS,
                  "schema_info": schema_info,
                  "data": csv_result,
                  "messages": [AIMessage(result)],
              }
  ```
  In each `except` branch, immediately before its `return`, add one audit call mirroring the existing status code (do not change the human-readable `error_type` strings). For the `TimeoutError` branch (after L361 `log(...)`):
  ```python
              _audit_logger.log_sql_exec(
                  sql=sql_query, dialect=dialect, row_count=None,
                  duration_ms=(_time.time() - start) * 1000.0, status=SQL_EXECUTE_TIMEOUT,
                  user_id=user_id, error=str(e),
              )
  ```
  Apply the analogous one-call insertion (with the branch's own status constant — `SQL_EXECUTE_TIMEOUT`, `SQL_SYNTAX_ERROR`, `SQL_UNKNOWN_ERROR`, `SQL_SECURITY_ERROR`, `execution_result`) right before each remaining `return` in the `OperationalError`, `SQLSecurityError`, and final `Exception` branches.

- [ ] **Step 12: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_execute_sql_audit.py -v`
  - Expected: PASS — 1 passed.

- [ ] **Step 13: Run the existing text2sql suite to confirm the 7+ error-string-coupled tests still pass.**
  - Run: `uv run pytest tests/test_text2sql_generate_sql.py tests/observability/ -v`
  - Expected: PASS — all green (human-readable `error_type` strings unchanged).

- [ ] **Step 14: Commit.**
  - Run: `git add openchatbi/observability/audit.py openchatbi/observability/callbacks.py openchatbi/text2sql/generate_sql.py tests/observability/test_audit.py tests/observability/test_callbacks.py tests/observability/test_execute_sql_audit.py && git commit -m "feat(observability): AuditLogger + ToolAuditCallback, wired into execute_sql_node (S1 Task 3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

### Task 4: Tracing integration + run-config builder + Config.observability submodel

**Files:**
- Create: `openchatbi/observability/tracing.py`
- Create: `tests/observability/test_tracing.py`
- Create: `tests/observability/test_tracing_contextvar_propagation.py`
- Modify: `openchatbi/config_loader.py` (add `ObservabilityConfig` submodel + `Config.observability` field, L26-89)
- Modify: `run_cli.py` (call `build_run_config`/`load_dotenv` at app start; config dict at L277)
- Modify: `sample_api/async_api.py` (config dict at L123; `load_dotenv` in `lifespan` L46)
- Modify: `sample_ui/streaming_ui.py` (config dict at L86)
- Modify: `sample_ui/streamlit_ui.py` (config dict at L71)
- Modify: `pyproject.toml` (add `observability` optional extra, after L106)

- [ ] **Step 1: Write failing test for `tracing.py`.** Create `tests/observability/test_tracing.py`:
  ```python
  """Tests for tracing callbacks + build_run_config."""

  from openchatbi.observability.tracing import build_run_config, get_tracing_callbacks


  def test_get_tracing_callbacks_disabled_returns_empty(monkeypatch) -> None:
      # No provider configured / disabled → empty list (zero-regression default).
      monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
      monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
      assert get_tracing_callbacks(enabled=False) == []


  def test_build_run_config_shape() -> None:
      cfg = build_run_config(user_id="alice", session_id="sess-1", request_id="req-1")
      assert cfg["configurable"]["thread_id"] == "alice-sess-1"
      assert cfg["configurable"]["user_id"] == "alice"
      assert isinstance(cfg["callbacks"], list)
      assert cfg["metadata"]["user_id"] == "alice"
      assert cfg["metadata"]["request_id"] == "req-1"
      assert cfg["run_name"]


  def test_build_run_config_preserves_base() -> None:
      base = {"configurable": {"thread_id": "existing-tid", "extra": 1}, "recursion_limit": 50}
      cfg = build_run_config(user_id="bob", session_id="s2", base=base)
      # base values survive; thread_id from base is preserved if already set.
      assert cfg["recursion_limit"] == 50
      assert cfg["configurable"]["extra"] == 1
      assert cfg["configurable"]["user_id"] == "bob"
  ```

- [ ] **Step 2: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_tracing.py -v`
  - Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.observability.tracing'`.

- [ ] **Step 3: Implement `tracing.py`.** Create `openchatbi/observability/tracing.py`:
  ```python
  """Tracing provider integration (Langfuse v3 self-hosted, LangSmith fallback).

  Credentials are read from environment / .env only (never from config files /
  git). When tracing is disabled or the provider lib is missing, returns ``[]``
  so the agent runs identically to today (zero regression).
  """

  from __future__ import annotations

  import os
  from copy import deepcopy
  from typing import Any

  from langchain_core.callbacks import BaseCallbackHandler

  from openchatbi.observability.callbacks import ToolAuditCallback
  from openchatbi.observability.context import set_run_context
  from openchatbi.utils import log


  def _resolve_observability_cfg() -> Any:
      try:
          from openchatbi import config as _cfg

          return getattr(_cfg.get(), "observability", None)
      except Exception:
          return None


  def get_tracing_callbacks(enabled: bool | None = None, provider: str | None = None) -> list[BaseCallbackHandler]:
      """Build provider tracing callbacks; ``[]`` when disabled / unavailable."""
      obs = _resolve_observability_cfg()
      if enabled is None:
          enabled = bool(getattr(getattr(obs, "tracing", None), "enabled", False))
      if not enabled:
          return []
      if provider is None:
          provider = getattr(getattr(obs, "tracing", None), "provider", None) or "langfuse"

      if provider == "langfuse":
          try:
              from langfuse.langchain import CallbackHandler  # Langfuse v3 path

              # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
              return [CallbackHandler()]
          except Exception as exc:
              log(f"Langfuse tracing unavailable: {exc!r}")
              return []
      if provider == "langsmith":
          try:
              from langchain_core.tracers import LangChainTracer

              if not os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
                  return []
              return [LangChainTracer()]
          except Exception as exc:
              log(f"LangSmith tracing unavailable: {exc!r}")
              return []
      return []


  def build_run_config(
      user_id: str,
      session_id: str,
      request_id: str | None = None,
      base: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
      """Build a LangGraph run config: configurable ids + tracing/audit callbacks + metadata.

      Also sets the run-context contextvars so deep code (execute_sql_node) can
      attribute work without signature threading.
      """
      set_run_context(user_id, request_id or f"{user_id}-{session_id}")

      cfg: dict[str, Any] = deepcopy(base) if base else {}
      configurable = dict(cfg.get("configurable") or {})
      configurable.setdefault("thread_id", f"{user_id}-{session_id}")
      configurable["user_id"] = user_id
      cfg["configurable"] = configurable

      callbacks: list[BaseCallbackHandler] = list(cfg.get("callbacks") or [])
      callbacks.append(ToolAuditCallback())
      callbacks.extend(get_tracing_callbacks())
      cfg["callbacks"] = callbacks

      metadata = dict(cfg.get("metadata") or {})
      metadata.update({"user_id": user_id, "session_id": session_id, "request_id": request_id})
      cfg["metadata"] = metadata

      cfg.setdefault("run_name", f"openchatbi:{user_id}:{session_id}")
      return cfg
  ```

- [ ] **Step 4: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_tracing.py -v`
  - Expected: PASS — 3 passed.

- [ ] **Step 5: Write failing test for contextvar propagation across the text2sql tool boundary.** Create `tests/observability/test_tracing_contextvar_propagation.py`:
  ```python
  """ContextVars set at turn start must survive the sync ToolNode / to_thread
  boundary that the text2sql tool crosses (get_sql_tools does not thread config
  to sql_graph.invoke at agent_graph.py:158/175)."""

  import asyncio
  import contextvars

  from openchatbi.observability.context import get_run_context, set_run_context


  def test_contextvar_survives_to_thread() -> None:
      set_run_context("alice", "req-7")

      def inner() -> tuple[str | None, str | None]:
          # Simulates execute_sql_node reading attribution inside the subgraph.
          return get_run_context()

      async def main() -> tuple[str | None, str | None]:
          # asyncio.to_thread copies the current context → ids must propagate.
          return await asyncio.to_thread(inner)

      assert asyncio.run(main()) == ("alice", "req-7")


  def test_contextvar_survives_copy_context() -> None:
      # LangGraph's sync ToolNode runs nodes via contextvars.copy_context().run.
      set_run_context("bob", "req-9")
      ctx = contextvars.copy_context()
      assert ctx.run(get_run_context) == ("bob", "req-9")
  ```

- [ ] **Step 6: Run the test (expect pass — this is the contextvar-propagation assertion the spec requires).**
  - Run: `uv run pytest tests/observability/test_tracing_contextvar_propagation.py -v`
  - Expected: PASS — 2 passed (confirms attribution survives the implicit-propagation boundary; no config threading needed for the text2sql tool).

- [ ] **Step 7: Write failing test for `Config.observability` submodel.** Create `tests/observability/test_config_observability.py`:
  ```python
  """Config.observability submodel is declared (pydantic extra='ignore' would
  otherwise silently drop it)."""

  from unittest.mock import MagicMock

  from openchatbi.config_loader import Config, ObservabilityConfig


  def test_observability_defaults_off() -> None:
      cfg = Config.from_dict({"default_llm": MagicMock()})
      assert isinstance(cfg.observability, ObservabilityConfig)
      assert cfg.observability.tracing.enabled is False
      assert cfg.observability.metrics.enabled is False
      assert cfg.observability.audit.enabled is False


  def test_observability_parsed_from_dict() -> None:
      cfg = Config.from_dict(
          {
              "default_llm": MagicMock(),
              "observability": {
                  "tracing": {"enabled": True, "provider": "langfuse"},
                  "metrics": {"enabled": True, "prometheus_port": 9100},
                  "audit": {"enabled": True, "mask_sql_literals": True},
              },
          }
      )
      assert cfg.observability.tracing.enabled is True
      assert cfg.observability.tracing.provider == "langfuse"
      assert cfg.observability.metrics.prometheus_port == 9100
      assert cfg.observability.audit.mask_sql_literals is True
  ```

- [ ] **Step 8: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_config_observability.py -v`
  - Expected: FAIL — `ImportError: cannot import name 'ObservabilityConfig'`.

- [ ] **Step 9: Add the `ObservabilityConfig` submodel and `Config.observability` field.** In `openchatbi/config_loader.py`, after the `LLMProviderConfig` class (before `class Config`, L24):
  ```python
  class TracingConfig(BaseModel):
      enabled: bool = False
      provider: str | None = None  # 'langfuse' | 'langsmith' | None
      langfuse_host: str | None = None
      sample_rate: float = 1.0


  class MetricsConfig(BaseModel):
      enabled: bool = False
      prometheus_port: int | None = None


  class AuditConfig(BaseModel):
      enabled: bool = False
      sink: str = "log"  # 'log' | 'file'
      path: str | None = None
      mask_sql_literals: bool = True


  class ObservabilityConfig(BaseModel):
      tracing: TracingConfig = TracingConfig()
      metrics: MetricsConfig = MetricsConfig()
      audit: AuditConfig = AuditConfig()
  ```
  Then add the field to `Config` after `context_config` (after L83):
  ```python
      # Observability Configuration (S1 — all sub-flags default OFF)
      observability: ObservabilityConfig = ObservabilityConfig()
  ```

- [ ] **Step 10: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_config_observability.py -v`
  - Expected: PASS — 2 passed.

- [ ] **Step 11: Add `load_dotenv()` + `build_run_config` at the CLI and API invoke sites.** In `run_cli.py`, after the `import sys` line (~L38) add:
  ```python
  try:
      from dotenv import load_dotenv

      load_dotenv()
  except ImportError:
      pass
  ```
  Replace the config dict at L277:
  ```python
      from openchatbi.observability.tracing import build_run_config

      config = build_run_config(user_id=args.user_id, session_id=args.session_id)
  ```
  In `sample_api/async_api.py`, in `lifespan` (after L47 docstring) add:
  ```python
      try:
          from dotenv import load_dotenv

          load_dotenv()
      except ImportError:
          pass
  ```
  Replace the config dict at L123:
  ```python
      from openchatbi.observability.tracing import build_run_config

      config = build_run_config(user_id=user_id, session_id=session_id)
  ```

- [ ] **Step 12: Add `build_run_config` at the two UI invoke sites.** In `sample_ui/streaming_ui.py`, replace the config dict at L86 with:
  ```python
      from openchatbi.observability.tracing import build_run_config

      config = build_run_config(user_id=user_id, session_id=session_id)
  ```
  In `sample_ui/streamlit_ui.py`, replace the config dict at L71 with the same two lines (`build_run_config(user_id=user_id, session_id=session_id)`). Do NOT touch `sample_ui/async_graph_manager.py` (build-only dead target, no invoke).

- [ ] **Step 13: Add the `observability` optional extra to `pyproject.toml`.** After the `test = [...]` block (after L106, before `dev = [`):
  ```toml
  observability = [
      "langfuse>=3,<4",
      "langsmith>=0.4.8,<1.0.0",
      "prometheus-client>=0.20.0,<1.0.0",
      "tiktoken>=0.7.0,<1.0.0",
      "python-dotenv>=1.0.0,<2.0.0",
  ]
  ```

- [ ] **Step 14: Run the tracing/config tests + CLI smoke import to confirm no regression.**
  - Run: `uv run pytest tests/observability/ tests/test_config_loader.py -v`
  - Expected: PASS — all green; `build_run_config` returns `callbacks=[ToolAuditCallback()]` and `get_tracing_callbacks()` returns `[]` by default.

- [ ] **Step 15: Commit.**
  - Run: `git add openchatbi/observability/tracing.py openchatbi/config_loader.py run_cli.py sample_api/async_api.py sample_ui/streaming_ui.py sample_ui/streamlit_ui.py pyproject.toml tests/observability/test_tracing.py tests/observability/test_tracing_contextvar_propagation.py tests/observability/test_config_observability.py && git commit -m "feat(observability): tracing callbacks + build_run_config + Config.observability, wired at 4 invoke sites (S1 Task 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

### Task 5: Per-turn token/cost streaming readout

**Files:**
- Modify: `openchatbi/streaming.py` (add `StreamUsage` to `StreamEvent` union, L51-81; add `turn_usage` accumulator + emit in `_process_message`, L165-202)
- Create: `tests/observability/test_streaming_usage.py`
- Modify: `run_cli.py` (render `StreamUsage` in `CliRenderer.render`, L131-179; NDJSON in `_emit_json`, L116-135)
- Modify: `sample_api/async_api.py` (serialize `StreamUsage` in `_event_to_dict`, L70-92)

- [ ] **Step 1: Write failing test for `StreamUsage` accumulation.** Create `tests/observability/test_streaming_usage.py`:
  ```python
  """AgentStreamProcessor aggregates per-turn token/cost into StreamUsage."""

  from langchain_core.messages import AIMessageChunk

  from openchatbi.streaming import AgentStreamProcessor, StreamUsage


  def _msg_event(chunk: AIMessageChunk, node: str) -> tuple:
      # Mirrors astream(stream_mode=["messages"]) triple: (namespace, "messages", (chunk, metadata)).
      return ((), "messages", (chunk, {"langgraph_node": node, "streaming_tokens": True}))


  def test_turn_usage_accumulates_from_final_chunk() -> None:
      processor = AgentStreamProcessor()
      chunk = AIMessageChunk(
          content="answer",
          usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
          response_metadata={"model_name": "gpt-4o"},
      )
      events = processor.process(*_msg_event(chunk, "llm_node"))
      # The token still streams; usage is folded into the accumulator.
      assert processor.turn_usage.turn_tokens == 120
      assert processor.turn_usage.by_model.get("gpt-4o") == 120
      assert processor.turn_usage.turn_cost_usd > 0.0

      usage_event = processor.emit_turn_usage()
      assert isinstance(usage_event, StreamUsage)
      assert usage_event.turn_tokens == 120


  def test_emit_turn_usage_none_when_no_usage() -> None:
      processor = AgentStreamProcessor()
      assert processor.emit_turn_usage() is None
  ```

- [ ] **Step 2: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_streaming_usage.py -v`
  - Expected: FAIL — `ImportError: cannot import name 'StreamUsage' from 'openchatbi.streaming'`.

- [ ] **Step 3: Add `StreamUsage` + accumulator to `streaming.py`.** In `openchatbi/streaming.py`, add the dataclass after `StreamInterrupt` (after L78), and extend the union:
  ```python
  @dataclass
  class StreamUsage:
      """Per-turn token / cost rollup, surfaced once at the end of a turn."""

      turn_tokens: int
      turn_cost_usd: float
      by_model: dict[str, int] = field(default_factory=dict)


  StreamEvent = StreamToken | StreamStep | StreamInterrupt | StreamUsage
  ```
  (Replace the existing `StreamEvent = StreamToken | StreamStep | StreamInterrupt` at L81.)
  Add the pricing import after the `get_text_from_message_chunk` import (after L24):
  ```python
  from openchatbi.observability.pricing import estimate_cost
  ```
  In `AgentStreamProcessor.__init__` (L165-167), add the accumulator:
  ```python
      def __init__(self) -> None:
          self._data_csv: Any = None
          self.final_response: str = ""
          self.turn_usage: StreamUsage = StreamUsage(turn_tokens=0, turn_cost_usd=0.0, by_model={})
  ```
  In `_process_message`, fold usage from the chunk's `usage_metadata` before the `if not token: return` guard so usage on the final (often empty-text) chunk is still captured. Insert right after `chunk, metadata = event_value[0], event_value[1]` and `node = metadata.get("langgraph_node")` (after L182):
  ```python
          usage = getattr(chunk, "usage_metadata", None)
          if usage:
              total = int(usage.get("total_tokens", 0) or 0)
              model_name = (getattr(chunk, "response_metadata", None) or {}).get("model_name", "") or "unknown"
              if total:
                  self.turn_usage.turn_tokens += total
                  self.turn_usage.by_model[model_name] = self.turn_usage.by_model.get(model_name, 0) + total
                  self.turn_usage.turn_cost_usd += estimate_cost(
                      model_name,
                      int(usage.get("input_tokens", 0) or 0),
                      int(usage.get("output_tokens", 0) or 0),
                  )
  ```
  Add an `emit_turn_usage` method to the class (after `process`, ~L176):
  ```python
      def emit_turn_usage(self) -> StreamUsage | None:
          """Return the accumulated per-turn usage, or None if nothing was recorded."""
          if self.turn_usage.turn_tokens <= 0:
              return None
          return self.turn_usage
  ```

- [ ] **Step 4: Run the test (expect pass).**
  - Run: `uv run pytest tests/observability/test_streaming_usage.py -v`
  - Expected: PASS — 2 passed.

- [ ] **Step 5: Write failing test for the CLI renderer + API serializer of `StreamUsage`.** Create `tests/observability/test_streaming_usage_render.py`:
  ```python
  """CLI renderer prints 'Turn: N tokens (~$X)'; async_api serializes StreamUsage."""

  from openchatbi.streaming import StreamUsage


  def test_cli_renders_turn_usage(capsys) -> None:
      from run_cli import CliRenderer

      renderer = CliRenderer(as_json=False, color=False)
      renderer.render(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
      out = capsys.readouterr().out
      assert "Turn: 120 tokens" in out
      assert "$0.0012" in out or "$0.001" in out


  def test_cli_json_usage(capsys) -> None:
      import json

      from run_cli import CliRenderer

      renderer = CliRenderer(as_json=True, color=False)
      renderer.render(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
      payload = json.loads(capsys.readouterr().out.strip())
      assert payload["type"] == "usage"
      assert payload["turn_tokens"] == 120


  def test_api_serializes_usage() -> None:
      from sample_api.async_api import _event_to_dict

      d = _event_to_dict(StreamUsage(turn_tokens=120, turn_cost_usd=0.0012, by_model={"gpt-4o": 120}))
      assert d["type"] == "usage"
      assert d["turn_tokens"] == 120
      assert d["by_model"] == {"gpt-4o": 120}
  ```

- [ ] **Step 6: Run the test (expect failure).**
  - Run: `uv run pytest tests/observability/test_streaming_usage_render.py -v`
  - Expected: FAIL — CLI renders nothing for `StreamUsage`; `_event_to_dict` returns `{"type": "unknown"}`.

- [ ] **Step 7: Render `StreamUsage` in `run_cli.py`.** Add the import to the `openchatbi.streaming` import block (after L46, alongside `StreamToken`):
  ```python
      StreamUsage,
  ```
  In `CliRenderer._emit_json` (L116-135), add a branch before the `else:  # StreamInterrupt`:
  ```python
          elif isinstance(event, StreamUsage):
              payload = {
                  "type": "usage",
                  "turn_tokens": event.turn_tokens,
                  "turn_cost_usd": event.turn_cost_usd,
                  "by_model": event.by_model,
              }
  ```
  In `CliRenderer.render` (L137-179), add a branch before the `elif isinstance(event, StreamInterrupt):`:
  ```python
          elif isinstance(event, StreamUsage):
              self._end_token_line()
              self._token_layer = None
              line = self._c(f"Turn: {event.turn_tokens} tokens (~${event.turn_cost_usd:.4f})", _Color.DIM)
              print(f"\n{line}")
  ```

- [ ] **Step 8: Emit `StreamUsage` at turn end in `run_cli.py` `_handle_state`.** In `_handle_state` (L210-...), before the final `renderer.final_answer(final)` call, render the usage rollup:
  ```python
      usage = processor.emit_turn_usage()
      if usage is not None:
          renderer.render(usage)
  ```
  (Place it after the interrupt early-return so usage is only shown on a completed turn.)

- [ ] **Step 9: Serialize `StreamUsage` in `async_api.py` + emit at turn end.** In `sample_api/async_api.py`, add `StreamUsage` to the `openchatbi.streaming` import block (after L20). In `_event_to_dict` (L70-92), add before the final `return {"type": "unknown"}`:
  ```python
      if isinstance(event, StreamUsage):
          return {
              "type": "usage",
              "turn_tokens": event.turn_tokens,
              "turn_cost_usd": event.turn_cost_usd,
              "by_model": event.by_model,
          }
  ```
  In `event_generator` (L143-159), after the `for event in processor.process(...)` loop completes and before emitting the interrupt/final_answer, emit usage:
  ```python
          usage = processor.emit_turn_usage()
          if usage is not None:
              yield json.dumps(_event_to_dict(usage), ensure_ascii=False) + "\n"
  ```

- [ ] **Step 10: Run the render tests (expect pass).**
  - Run: `uv run pytest tests/observability/test_streaming_usage_render.py -v`
  - Expected: PASS — 3 passed.

- [ ] **Step 11: Run the full streaming + observability suite to confirm no regression.**
  - Run: `uv run pytest tests/test_streaming.py tests/observability/ -v`
  - Expected: PASS — all green; existing `StreamEvent` consumers unaffected (union extended, not changed).

- [ ] **Step 12: Commit.**
  - Run: `git add openchatbi/streaming.py run_cli.py sample_api/async_api.py tests/observability/test_streaming_usage.py tests/observability/test_streaming_usage_render.py && git commit -m "feat(observability): per-turn StreamUsage readout (tokens + cost) in CLI/API (S1 Task 5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

## S2-S3-substrates(Tasks 6-7)

### Task 6: S2 — SQL quality evaluator (`SimpleSQLEvaluator` + `ConfidenceResult`)

**Files:**
- Create: `openchatbi/text2sql/confidence.py`
- Create: `tests/text2sql/test_confidence.py`

Steps:

- [ ] **Step 1: Write failing test for `ConfidenceResult` dataclass shape.**
  Create `tests/text2sql/test_confidence.py` with the imports and a first test asserting the dataclass fields exist with the contract names:
  ```python
  """Tests for the S2 SQL quality evaluator (SimpleSQLEvaluator)."""

  import json

  import pytest
  from langchain_core.language_models import FakeListChatModel

  from openchatbi.text2sql.confidence import ConfidenceResult, SimpleSQLEvaluator


  def test_confidence_result_fields():
      result = ConfidenceResult(
          score=0.83,
          reasons=["WHERE clause matches the date filter"],
          checks={
              "select_columns": True,
              "where": True,
              "calc": True,
              "subquery": True,
              "joins": True,
              "exec_result": True,
          },
      )
      assert result.score == 0.83
      assert result.reasons == ["WHERE clause matches the date filter"]
      assert result.checks["select_columns"] is True
      assert set(result.checks) == {
          "select_columns",
          "where",
          "calc",
          "subquery",
          "joins",
          "exec_result",
      }
  ```

- [ ] **Step 2: Run the test — Expected: FAIL (ModuleNotFoundError: openchatbi.text2sql.confidence).**
  Run: `uv run pytest tests/text2sql/test_confidence.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.text2sql.confidence'`.

- [ ] **Step 3: Create `openchatbi/text2sql/confidence.py` with the dataclass + evaluator skeleton.**
  Single structured-output LLM call running the Dataherald 6-step rubric. Default `llm = get_default_llm()`, low temperature via `model_kwargs` bind when supported (graceful no-op otherwise).
  ```python
  """S2 — SQL quality evaluator (single structured-output LLM call, 6-step rubric).

  Reused by HITL confidence gate, the Eval LLM-as-Judge, and the Memory
  promotion gate. Default behaviour is OFF at every call site; this module only
  computes a score when explicitly invoked.
  """

  from __future__ import annotations

  import json
  from dataclasses import dataclass, field

  from langchain_core.language_models import BaseChatModel
  from langchain_core.messages import HumanMessage, SystemMessage

  from openchatbi.llm.llm import get_default_llm
  from openchatbi.utils import log

  # Ordered rubric check keys (Dataherald 6-step rubric).
  RUBRIC_CHECKS: tuple[str, ...] = (
      "select_columns",  # SELECT columns map to the question's requested fields
      "where",           # WHERE conditions correctly express the filters
      "calc",            # calculations / aggregations are correct
      "subquery",        # subqueries are correctly decomposed
      "joins",           # JOIN columns match across tables
      "exec_result",     # the (sampled) execution result is plausible
  )

  _RUBRIC_PROMPT = """You are a strict SQL reviewer. Score whether the SQL correctly answers the question.
  Apply these six checks, each strictly true or false:
  1. select_columns: the SELECT columns map to the fields the question asks for.
  2. where: the WHERE conditions correctly express every filter implied by the question.
  3. calc: aggregations and arithmetic are correct.
  4. subquery: any subqueries are correctly decomposed and necessary.
  5. joins: JOIN keys match the correct columns across tables.
  6. exec_result: the sampled execution result (if any) is plausible for the question.

  Schema info:
  {schema_info}

  Data sample (may be empty):
  {data_sample}

  Question:
  {question}

  SQL:
  {sql}

  Respond with ONLY a JSON object, no prose, of the exact form:
  {{"score": <float 0..1>, "reasons": [<string>, ...], "checks": {{"select_columns": <bool>, "where": <bool>, "calc": <bool>, "subquery": <bool>, "joins": <bool>, "exec_result": <bool>}}}}
  """


  @dataclass
  class ConfidenceResult:
      score: float
      reasons: list[str] = field(default_factory=list)
      checks: dict[str, bool] = field(default_factory=dict)


  class SimpleSQLEvaluator:
      """Single LLM call that scores SQL quality against a 6-step rubric."""

      def __init__(self, llm: BaseChatModel | None = None):
          self.llm = llm if llm is not None else get_default_llm()

      def _low_temp_llm(self) -> BaseChatModel:
          # Bind a low temperature when the provider supports it; no-op otherwise.
          try:
              return self.llm.bind(temperature=0.0)
          except Exception:
              return self.llm

      def evaluate(
          self,
          question: str,
          sql: str,
          schema_info: dict,
          data_sample: str | None,
      ) -> ConfidenceResult:
          prompt = _RUBRIC_PROMPT.format(
              schema_info=json.dumps(schema_info, default=str),
              data_sample=data_sample or "",
              question=question,
              sql=sql,
          )
          messages = [
              SystemMessage(content="You are a precise SQL correctness evaluator."),
              HumanMessage(content=prompt),
          ]
          try:
              response = self._low_temp_llm().invoke(messages)
              return self._parse(getattr(response, "content", str(response)))
          except Exception as exc:  # never raise into the calling graph
              log(f"SimpleSQLEvaluator.evaluate failed: {exc}")
              return ConfidenceResult(score=0.0, reasons=[f"evaluator error: {exc}"], checks={})

      @staticmethod
      def _parse(content: str) -> ConfidenceResult:
          text = content.strip()
          # Tolerate fenced code blocks around the JSON payload.
          if text.startswith("```"):
              text = text.strip("`")
              if "\n" in text:
                  text = text.split("\n", 1)[1]
          start, end = text.find("{"), text.rfind("}")
          if start == -1 or end == -1:
              return ConfidenceResult(score=0.0, reasons=["unparseable evaluator output"], checks={})
          data = json.loads(text[start : end + 1])
          checks = {k: bool(data.get("checks", {}).get(k, False)) for k in RUBRIC_CHECKS}
          score = float(data.get("score", 0.0))
          score = max(0.0, min(1.0, score))
          reasons = [str(r) for r in data.get("reasons", [])]
          return ConfidenceResult(score=score, reasons=reasons, checks=checks)
  ```

- [ ] **Step 4: Run the dataclass test — Expected: PASS.**
  Run: `uv run pytest tests/text2sql/test_confidence.py::test_confidence_result_fields -v`
  Expected: PASS.

- [ ] **Step 5: Write failing test for `evaluate()` parsing a structured verdict from `mock_llm`.**
  Append to `tests/text2sql/test_confidence.py`. `FakeListChatModel` returns the next canned string from `responses` on each `invoke`; we feed it the rubric JSON so the parser is exercised end-to-end without a real LLM.
  ```python
  def _verdict_json(score: float, all_true: bool = True) -> str:
      return json.dumps(
          {
              "score": score,
              "reasons": ["columns match", "where matches"],
              "checks": {
                  "select_columns": all_true,
                  "where": all_true,
                  "calc": all_true,
                  "subquery": all_true,
                  "joins": all_true,
                  "exec_result": all_true,
              },
          }
      )


  def test_evaluate_parses_structured_verdict():
      mock_llm = FakeListChatModel(responses=[_verdict_json(0.92)])
      evaluator = SimpleSQLEvaluator(llm=mock_llm)
      result = evaluator.evaluate(
          question="How many users are there?",
          sql="SELECT COUNT(*) FROM users;",
          schema_info={"users": ["id", "name"]},
          data_sample="count\n42",
      )
      assert isinstance(result, ConfidenceResult)
      assert result.score == 0.92
      assert result.checks["joins"] is True
      assert all(result.checks[k] for k in result.checks)
      assert "columns match" in result.reasons


  def test_evaluate_clamps_score_and_handles_false_checks():
      mock_llm = FakeListChatModel(responses=[_verdict_json(1.7, all_true=False)])
      evaluator = SimpleSQLEvaluator(llm=mock_llm)
      result = evaluator.evaluate("q", "SELECT 1", {}, None)
      assert result.score == 1.0  # clamped into [0, 1]
      assert result.checks["where"] is False


  def test_evaluate_never_raises_on_bad_output():
      mock_llm = FakeListChatModel(responses=["not json at all"])
      evaluator = SimpleSQLEvaluator(llm=mock_llm)
      result = evaluator.evaluate("q", "SELECT 1", {}, None)
      assert result.score == 0.0
      assert result.checks == {}
  ```

- [ ] **Step 6: Run the evaluate tests — Expected: PASS.**
  Run: `uv run pytest tests/text2sql/test_confidence.py -v`
  Expected: PASS — all four tests green (`mock_llm` `FakeListChatModel` returns the canned JSON verdict, the parser clamps the score and tolerates non-JSON).

- [ ] **Step 7: Commit.**
  Run:
  ```
  git checkout -b feat/harness-s2-s3-substrates
  git add openchatbi/text2sql/confidence.py tests/text2sql/test_confidence.py
  git commit -m "$(cat <<'EOF'
  feat(text2sql): add S2 SimpleSQLEvaluator confidence module

  Single structured-output LLM call running the Dataherald 6-step rubric
  (select_columns/where/calc/subquery/joins/exec_result). Defaults to
  get_default_llm(), low temperature, never raises into the calling graph.
  Reused by HITL gate, Eval judge, and Memory promotion gate. Inert until a
  caller invokes it (all feature flags default OFF).

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 7: S3 — runtime-mutable learned SQL store (`LearnedSQLStore` + 3-tuple retriever)

**Files:**
- Modify: `openchatbi/text2sql/text2sql_utils.py` (`_init_sql_example_retriever` → 3-tuple; add `LearnedSQLStore`)
- Modify: `openchatbi/text2sql/data.py:16` (module-level unpacking → 3-tuple)
- Create: `tests/text2sql/test_learned_sql_store.py`

Steps:

- [ ] **Step 1: Write failing test for the 3-tuple return of `_init_sql_example_retriever`.**
  Create `tests/text2sql/test_learned_sql_store.py`. The `mock_catalog_store` fixture has no SQL examples (`get_sql_examples()` returns `[]`), and `get_embedding_model()` is unconfigured in test mode, so `create_vector_db` falls back to `SimpleStore` (BM25, no embedding). We bypass real config by passing the store handle directly.
  ```python
  """Tests for the S3 runtime-mutable learned SQL store (LearnedSQLStore)."""

  import threading

  import pytest

  from openchatbi.text2sql.text2sql_utils import LearnedSQLStore, _init_sql_example_retriever
  from openchatbi.utils import SimpleStore


  def test_init_sql_example_retriever_returns_three_tuple(mock_catalog_store, monkeypatch):
      # Force the no-embedding (SimpleStore/BM25) path.
      monkeypatch.setattr(
          "openchatbi.text2sql.text2sql_utils.get_embedding_model", lambda: None
      )
      result = _init_sql_example_retriever(mock_catalog_store, vector_db_path=None)
      assert isinstance(result, tuple)
      assert len(result) == 3
      retriever, example_dict, vector_db = result
      assert isinstance(example_dict, dict)
      assert isinstance(vector_db, SimpleStore)
      assert hasattr(retriever, "invoke")
  ```

- [ ] **Step 2: Run the test — Expected: FAIL.**
  Run: `uv run pytest tests/text2sql/test_learned_sql_store.py::test_init_sql_example_retriever_returns_three_tuple -v`
  Expected: FAIL — `ImportError: cannot import name 'LearnedSQLStore'` (module currently returns a 2-tuple and has no `LearnedSQLStore`).

- [ ] **Step 3: Change `_init_sql_example_retriever` to a 3-tuple and add `LearnedSQLStore`.**
  Replace the body of `openchatbi/text2sql/text2sql_utils.py` (lines 1-31, the `_init_sql_example_retriever` function and module imports) with the version below. Note: `add_texts` rebuilds BM25 `O(N)` and is non-threadsafe (utils.py:457), so every mutation is guarded by `self.lock`; `retrieve` uses MMR (works on both Chroma `as_retriever`/`max_marginal_relevance_search` and `SimpleStore.max_marginal_relevance_search`, utils.py:532). The dict mirror keeps `_get_relevant_sql_examples_prompt` (generate_sql.py:204) lookups working in place.
  ```python
  """Utility functions for text2sql retrieval systems."""

  import threading
  from datetime import datetime, timezone

  from openchatbi.llm.llm import get_embedding_model
  from openchatbi.utils import create_vector_db, log


  def _init_sql_example_retriever(catalog, vector_db_path: str = None):
      """Initialize SQL example retriever from catalog.

      Args:
          catalog: Catalog store containing SQL examples.
          vector_db_path: Path to the vector database file.

      Returns:
          tuple: (retriever, sql_example_dict, vector_db)
      """
      sql_examples = catalog.get_sql_examples()
      sql_example_dict = {q: (sql, table) for q, sql, table in sql_examples}

      texts = list(sql_example_dict.keys())
      vector_db = create_vector_db(
          texts,
          get_embedding_model(),
          collection_name="text2sql",
          collection_metadata={"hnsw:space": "cosine"},
          chroma_db_path=vector_db_path,
      )
      retriever = vector_db.as_retriever(
          search_type="mmr", search_kwargs={"distance_metric": "cosine", "fetch_k": 30, "k": 10}
      )
      return retriever, sql_example_dict, vector_db


  class LearnedSQLStore:
      """Runtime-mutable learned SQL knowledge base.

      Wraps the text2sql vector store so that approved (``source='golden'``) and
      auto-captured (``source='auto'``) examples can be written at runtime and
      retrieved alongside the static catalog examples. Writes are guarded by a
      lock because ``SimpleStore.add_texts`` rebuilds the BM25 index O(N) and is
      not threadsafe; callers are responsible for the durable YAML half of the
      dual-write contract (the in-memory ``add_texts`` here is the volatile half).
      """

      def __init__(self, vector_db, example_dict: dict, lock: threading.Lock | None = None):
          self.vector_db = vector_db
          self.example_dict = example_dict
          self.lock = lock or threading.Lock()

      def add(
          self,
          question: str,
          sql: str,
          tables: list[str],
          *,
          source: str,
          importance: float = 1.0,
          namespace: str = "global",
      ) -> None:
          """Add a learned example to the runtime store (volatile half of dual-write).

          Args:
              question: Natural-language question (the indexed text).
              sql: SQL answer.
              tables: Tables used by the SQL.
              source: Provenance; 'golden' (human-approved) or 'auto' (S2-gated capture).
              importance: Base importance weight used by composite scoring.
              namespace: Tenant/scope tag; 'global' must hold only schema-level patterns.
          """
          now = datetime.now(timezone.utc).isoformat()
          metadata = {
              "sql": sql,
              "tables": tables,
              "source": source,
              "importance": importance,
              "use_count": 0,
              "last_used": now,
              "namespace": namespace,
          }
          with self.lock:
              # Volatile half: BM25 rebuild / Chroma add is O(N) and non-threadsafe.
              self.vector_db.add_texts([question], metadatas=[metadata])
              # Mirror into the dict so _get_relevant_sql_examples_prompt keeps working.
              self.example_dict[question] = (sql, tables)

      def add_golden_sql(self, question: str, sql: str, tables: list[str]) -> None:
          """Alias: add a human-approved golden example with high importance."""
          self.add(question, sql, tables, source="golden", importance=2.0)

      def retrieve(
          self,
          question: str,
          k: int = 10,
          score_fn=None,
      ) -> list[tuple[str, str, list[str]]]:
          """Retrieve top-k learned examples for a question.

          Args:
              question: Query text.
              k: Number of examples to return.
              score_fn: Optional re-ranker ``(metadata, base_rank) -> float`` (e.g.
                  composite_score from memory_scoring); higher is better. When None,
                  the underlying MMR order is preserved.

          Returns:
              List of (question, sql, tables) tuples.
          """
          docs = self.vector_db.max_marginal_relevance_search(question, k=max(k, 1), fetch_k=30)
          ranked = list(enumerate(docs))
          if score_fn is not None:
              ranked.sort(key=lambda pair: score_fn(pair[1].metadata, pair[0]), reverse=True)
          results: list[tuple[str, str, list[str]]] = []
          for _, doc in ranked[:k]:
              q = doc.page_content
              sql = doc.metadata.get("sql")
              if sql is None and q in self.example_dict:
                  sql, tables = self.example_dict[q]
              else:
                  tables = doc.metadata.get("tables", [])
              results.append((q, sql, tables))
          return results
  ```

- [ ] **Step 4: Run the 3-tuple test — Expected: PASS.**
  Run: `uv run pytest tests/text2sql/test_learned_sql_store.py::test_init_sql_example_retriever_returns_three_tuple -v`
  Expected: PASS.

- [ ] **Step 5: Update `data.py` module-level unpacking to the 3-tuple.**
  In `openchatbi/text2sql/data.py`, expose the `vector_db` handle and build the shared `LearnedSQLStore` + lock so HITL/Memory callers reuse one store. Edit lines 4, 16, and 21.
  Change the import on line 4:
  ```python
  from openchatbi.text2sql.text2sql_utils import (
      LearnedSQLStore,
      _init_sql_example_retriever,
      _init_table_selection_example_dict,
  )
  ```
  Change the `if _catalog_store:` branch (line 16) and `else` branch (line 21):
  ```python
  if _catalog_store:
      sql_example_retriever, sql_example_dicts, sql_example_vector_db = _init_sql_example_retriever(
          _catalog_store, config.get().vector_db_path
      )
      learned_sql_store = LearnedSQLStore(sql_example_vector_db, sql_example_dicts, threading.Lock())
      table_selection_retriever, table_selection_example_dict = _init_table_selection_example_dict(
          _catalog_store, config.get().vector_db_path
      )
  else:
      sql_example_retriever, sql_example_dicts, sql_example_vector_db = None, {}, None
      learned_sql_store = None
      table_selection_retriever, table_selection_example_dict = None, {}
  ```
  Add `import threading` at the top of `data.py` (after `import os`).

- [ ] **Step 6: Write failing test for `add`→`retrieve` round-trip + threadsafe `add` on the SimpleStore path.**
  Append to `tests/text2sql/test_learned_sql_store.py`. Build a `LearnedSQLStore` over a `SimpleStore` (no-embedding BM25) seeded so BM25 is non-empty, then verify golden add is retrievable, namespace metadata is stamped, the dict mirror is updated, and concurrent adds don't corrupt the index.
  ```python
  def _make_store():
      vector_db = SimpleStore(
          ["How many users are there?"],
          metadatas=[{"sql": "SELECT COUNT(*) FROM users;", "tables": ["users"], "source": "static"}],
      )
      example_dict = {"How many users are there?": ("SELECT COUNT(*) FROM users;", ["users"])}
      return LearnedSQLStore(vector_db, example_dict, threading.Lock())


  def test_add_then_retrieve_round_trip_simplestore():
      store = _make_store()
      store.add_golden_sql(
          "What is the average age of users?",
          "SELECT AVG(age) FROM users;",
          ["users"],
      )
      # dict mirror updated in place
      assert store.example_dict["What is the average age of users?"] == (
          "SELECT AVG(age) FROM users;",
          ["users"],
      )
      results = store.retrieve("average age users", k=5)
      questions = [q for q, _, _ in results]
      assert "What is the average age of users?" in questions
      sql = dict((q, s) for q, s, _ in results)["What is the average age of users?"]
      assert sql == "SELECT AVG(age) FROM users;"


  def test_add_stamps_namespace_and_source_metadata():
      store = _make_store()
      store.add("foo bar baz", "SELECT 1", ["t"], source="auto", importance=0.5, namespace="tenant_a")
      meta = next(d.metadata for d in store.vector_db.documents if d.page_content == "foo bar baz")
      assert meta["source"] == "auto"
      assert meta["namespace"] == "tenant_a"
      assert meta["importance"] == 0.5
      assert meta["use_count"] == 0
      assert "last_used" in meta


  def test_concurrent_add_is_threadsafe():
      store = _make_store()

      def worker(i):
          store.add(f"question number {i}", f"SELECT {i}", ["t"], source="auto")

      threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
      for t in threads:
          t.start()
      for t in threads:
          t.join()
      # 1 seed + 20 concurrent adds, no lost writes and BM25 index consistent.
      assert len(store.vector_db.texts) == 21
      assert len(store.vector_db.documents) == 21
      assert len(store.vector_db.tokenized_corpus) == 21
      results = store.retrieve("question number 7", k=21)
      assert "question number 7" in [q for q, _, _ in results]


  def test_retrieve_score_fn_reranks():
      store = _make_store()
      store.add("alpha query about users", "SELECT a", ["users"], source="golden", importance=2.0)
      store.add("beta query about users", "SELECT b", ["users"], source="auto", importance=0.1)

      # score_fn prefers higher importance regardless of MMR order.
      def score_fn(meta, base_rank):
          return meta.get("importance", 0.0)

      results = store.retrieve("users query", k=3, score_fn=score_fn)
      assert results[0][0] == "alpha query about users"
  ```

- [ ] **Step 7: Run the round-trip + threadsafety tests — Expected: PASS.**
  Run: `uv run pytest tests/text2sql/test_learned_sql_store.py -v`
  Expected: PASS — all tests green (`add_golden_sql` round-trips through BM25, namespace/source/use_count/last_used metadata is stamped, 20 concurrent adds yield exactly 21 consistent entries, `score_fn` re-ranks by importance).

- [ ] **Step 8: Run the existing text2sql data/import suite to confirm the 3-tuple change is non-regressive.**
  Run: `uv run pytest tests/text2sql/ -v`
  Expected: PASS — the `data.py` module-level unpacking now matches the 3-tuple return; no existing caller of `sql_example_retriever`/`sql_example_dicts` changed (both remain at the same positions), so `_get_relevant_sql_examples_prompt` is unaffected.

- [ ] **Step 9: Commit.**
  Run:
  ```
  git add openchatbi/text2sql/text2sql_utils.py openchatbi/text2sql/data.py tests/text2sql/test_learned_sql_store.py
  git commit -m "$(cat <<'EOF'
  feat(text2sql): S3 runtime-mutable LearnedSQLStore + 3-tuple retriever

  _init_sql_example_retriever now returns (retriever, dict, vector_db) so the
  vector store handle is reachable for runtime writes. Add LearnedSQLStore
  unifying golden (human-approved) and auto (S2-gated) examples in one store:
  - lock-guarded add() because SimpleStore.add_texts rebuilds BM25 O(N) and is
    non-threadsafe; works on both Chroma and SimpleStore/BM25 paths
  - source/importance/use_count/last_used/namespace metadata
  - add_golden_sql alias and retrieve(score_fn) for composite re-ranking
  In-memory add is the volatile half of the dual-write contract; callers persist
  YAML. data.py builds one shared store + lock. No behaviour change yet (no
  caller wired; all flags default OFF).

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Error(Tasks 8-10)

### Task 8: Create `openchatbi/text2sql/errors.py` — RecoveryStrategy enum, Text2SQLError base + 6 subclasses, classify_sql_exception

**Files:**
- Create: `openchatbi/text2sql/errors.py`
- Create: `tests/test_text2sql_errors.py`
- Modify: `openchatbi/text2sql/generate_sql.py` (re-export `SQLSecurityError` from new module for back-compat; lines 31-33 define it today, line 14-23 imports block)

- [ ] **Step 1: Write the failing test for the error taxonomy.**
  Create `tests/test_text2sql_errors.py` with the full taxonomy + classifier coverage. These mirror the exact exception shapes the existing `execute_sql_node` tests raise (`OperationalError` with timeout/syntax/other messages, `ProgrammingError`, `DatabaseError`, `SQLSecurityError`).

  ```python
  """Tests for the structured Text2SQL error taxonomy and classifier."""

  import pytest
  from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError, TimeoutError

  from openchatbi.constants import (
      SQL_EXECUTE_TIMEOUT,
      SQL_NA,
      SQL_SECURITY_ERROR,
      SQL_SYNTAX_ERROR,
      SQL_UNKNOWN_ERROR,
  )
  from openchatbi.text2sql.errors import (
      DBTimeoutError,
      EmptyResultError,
      InvalidDBConnectionError,
      RecoveryStrategy,
      SQLSecurityError,
      SQLSyntaxError,
      Text2SQLError,
      UnknownSQLError,
      classify_sql_exception,
  )


  class TestRecoveryStrategy:
      def test_enum_values(self):
          assert RecoveryStrategy.RETRY == "retry"
          assert RecoveryStrategy.RETRY_WITH_NEW_TABLE == "retry_with_new_table"
          assert RecoveryStrategy.SURFACE_TO_USER == "surface_to_user"
          assert RecoveryStrategy.ABORT == "abort"

      def test_is_str_enum(self):
          assert isinstance(RecoveryStrategy.RETRY, str)


  class TestText2SQLErrorSubclasses:
      def test_base_fields(self):
          orig = ValueError("boom")
          err = Text2SQLError(
              "msg",
              code=SQL_UNKNOWN_ERROR,
              recovery_strategy=RecoveryStrategy.RETRY,
              user_message="please retry",
              orig=orig,
          )
          assert err.code == SQL_UNKNOWN_ERROR
          assert err.recovery_strategy is RecoveryStrategy.RETRY
          assert err.user_message == "please retry"
          assert err.orig is orig

      def test_security_error_defaults(self):
          err = SQLSecurityError("Operation not allowed")
          assert err.code == SQL_SECURITY_ERROR
          assert err.recovery_strategy is RecoveryStrategy.SURFACE_TO_USER
          assert isinstance(err, Text2SQLError)

      def test_syntax_error_defaults(self):
          err = SQLSyntaxError("bad syntax")
          assert err.code == SQL_SYNTAX_ERROR
          assert err.recovery_strategy is RecoveryStrategy.RETRY

      def test_invalid_connection_defaults(self):
          err = InvalidDBConnectionError("bad creds")
          assert err.code == SQL_EXECUTE_TIMEOUT
          assert err.recovery_strategy is RecoveryStrategy.SURFACE_TO_USER

      def test_db_timeout_defaults(self):
          err = DBTimeoutError("timed out")
          assert err.code == SQL_EXECUTE_TIMEOUT
          assert err.recovery_strategy is RecoveryStrategy.ABORT

      def test_empty_result_defaults(self):
          err = EmptyResultError("no rows")
          assert err.code == SQL_NA   # 见约定#6:EmptyResultError.code = SQL_NA
          assert err.recovery_strategy is RecoveryStrategy.RETRY_WITH_NEW_TABLE

      def test_unknown_error_defaults(self):
          err = UnknownSQLError("disk i/o error")
          assert err.code == SQL_UNKNOWN_ERROR
          assert err.recovery_strategy is RecoveryStrategy.RETRY


  class TestClassifySqlException:
      def test_security_error_passthrough(self):
          out = classify_sql_exception(SQLSecurityError("Operation not allowed: x"))
          assert isinstance(out, SQLSecurityError)
          assert out.code == SQL_SECURITY_ERROR
          assert out.error_type == "SQL security error"

      def test_timeout_error(self):
          out = classify_sql_exception(TimeoutError("query timed out"))
          assert isinstance(out, DBTimeoutError)
          assert out.code == SQL_EXECUTE_TIMEOUT
          assert out.error_type == "Database connection timeout"

      def test_operational_timeout_or_connection(self):
          out = classify_sql_exception(OperationalError("", {}, Exception("connection refused")))
          assert isinstance(out, DBTimeoutError)
          assert out.code == SQL_EXECUTE_TIMEOUT

      def test_operational_syntax(self):
          out = classify_sql_exception(OperationalError("", {}, Exception('near "<": syntax error')))
          assert isinstance(out, SQLSyntaxError)
          assert out.code == SQL_SYNTAX_ERROR
          assert out.error_type == "SQL syntax error"

      def test_operational_other_is_operational(self):
          out = classify_sql_exception(OperationalError("", {}, Exception("disk i/o error")))
          assert out.code == SQL_UNKNOWN_ERROR
          assert out.error_type == "Database operational error"

      def test_programming_error_is_syntax(self):
          out = classify_sql_exception(ProgrammingError("", "", "Syntax error"))
          assert isinstance(out, SQLSyntaxError)
          assert out.code == SQL_SYNTAX_ERROR
          assert out.error_type == "SQL syntax error"

      def test_database_error_is_unknown(self):
          out = classify_sql_exception(DatabaseError("", "", "generic db error"))
          assert out.code == SQL_UNKNOWN_ERROR
          assert out.error_type == "Database error"

      def test_generic_exception_is_unknown(self):
          out = classify_sql_exception(RuntimeError("something else"))
          assert isinstance(out, UnknownSQLError)
          assert out.code == SQL_UNKNOWN_ERROR
          assert out.error_type == "Unexpected error"

      def test_orig_is_preserved(self):
          src = ProgrammingError("", "", "Syntax error")
          out = classify_sql_exception(src)
          assert out.orig is src
  ```

- [ ] **Step 2: Run the test (expect failure — module does not exist yet).**
  Run: `uv run pytest tests/test_text2sql_errors.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'openchatbi.text2sql.errors'`.

- [ ] **Step 3: Implement `openchatbi/text2sql/errors.py`.**
  Reuse `_extract_exception_message` / `_classify_operational_error` from `generate_sql.py` via a local import inside `classify_sql_exception` to avoid a circular import (`generate_sql` will import `errors`). The `error_type` attribute carries the EXACT human-readable strings the existing tests assert on so the node can reuse them.

  ```python
  """Structured error taxonomy and classifier for the Text2SQL subgraph.

  Keeps the SQL_* status codes and the human-readable error_type strings that
  downstream tests are coupled to; new structured information (recovery strategy,
  error class, error code) is carried alongside without mutating those strings.
  """

  from enum import Enum

  from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError, TimeoutError

  from openchatbi.constants import (
      SQL_EXECUTE_TIMEOUT,
      SQL_NA,
      SQL_SECURITY_ERROR,
      SQL_SYNTAX_ERROR,
      SQL_UNKNOWN_ERROR,
  )


  class RecoveryStrategy(str, Enum):
      """How the graph should react to a classified Text2SQL error."""

      RETRY = "retry"
      RETRY_WITH_NEW_TABLE = "retry_with_new_table"
      SURFACE_TO_USER = "surface_to_user"
      ABORT = "abort"


  class Text2SQLError(Exception):
      """Base structured error for the Text2SQL subgraph.

      Attributes:
          code: One of the existing SQL_* status code constants (downstream-compatible).
          recovery_strategy: How the graph should recover (see RecoveryStrategy).
          error_type: Human-readable label preserved for legacy test/UI coupling.
          user_message: Optional message safe to surface to the end user.
          orig: The originating exception, if any.
      """

      code: str = SQL_UNKNOWN_ERROR
      recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
      error_type: str = "Unexpected error"

      def __init__(
          self,
          message: str = "",
          *,
          code: str | None = None,
          recovery_strategy: RecoveryStrategy | None = None,
          error_type: str | None = None,
          user_message: str | None = None,
          orig: BaseException | None = None,
      ) -> None:
          super().__init__(message)
          if code is not None:
              self.code = code
          if recovery_strategy is not None:
              self.recovery_strategy = recovery_strategy
          if error_type is not None:
              self.error_type = error_type
          self.user_message = user_message
          self.orig = orig


  class SQLSecurityError(Text2SQLError):
      """Raised when generated SQL fails safety validation."""

      code = SQL_SECURITY_ERROR
      recovery_strategy = RecoveryStrategy.SURFACE_TO_USER
      error_type = "SQL security error"


  class SQLSyntaxError(Text2SQLError):
      """Raised when the database reports a SQL syntax/parse error."""

      code = SQL_SYNTAX_ERROR
      recovery_strategy = RecoveryStrategy.RETRY
      error_type = "SQL syntax error"


  class InvalidDBConnectionError(Text2SQLError):
      """Raised when the database connection itself is invalid/unauthorized."""

      code = SQL_EXECUTE_TIMEOUT
      recovery_strategy = RecoveryStrategy.SURFACE_TO_USER
      error_type = "Database connection error"


  class DBTimeoutError(Text2SQLError):
      """Raised on database timeout / dropped connection during execution."""

      code = SQL_EXECUTE_TIMEOUT
      recovery_strategy = RecoveryStrategy.ABORT
      error_type = "Database connection timeout"


  class EmptyResultError(Text2SQLError):
      """Raised (opt-in only) when a query returns zero rows."""

      code = SQL_NA   # 见约定#6:软失败码,非 SQL_SUCCESS;默认 gate 关闭时根本不构造此异常
      recovery_strategy = RecoveryStrategy.RETRY_WITH_NEW_TABLE
      error_type = "Empty result"


  class UnknownSQLError(Text2SQLError):
      """Catch-all for operational/database/unexpected errors."""

      code = SQL_UNKNOWN_ERROR
      recovery_strategy = RecoveryStrategy.RETRY
      error_type = "Unexpected error"


  def classify_sql_exception(exc: BaseException) -> Text2SQLError:
      """Classify a raw exception into a structured Text2SQLError.

      Reuses the existing message extraction and operational-error heuristics so
      classification stays consistent with the legacy multi-branch except chain.
      The returned error's ``error_type`` matches the exact human-readable strings
      that downstream tests assert on.
      """
      from openchatbi.text2sql.generate_sql import (
          _classify_operational_error,
          _extract_exception_message,
      )

      if isinstance(exc, Text2SQLError):
          return exc

      if isinstance(exc, TimeoutError):
          return DBTimeoutError(_extract_exception_message(exc), orig=exc)

      if isinstance(exc, OperationalError):
          category = _classify_operational_error(exc)
          if category == "timeout_or_connection":
              return DBTimeoutError(str(exc), orig=exc)
          if category == "syntax":
              return SQLSyntaxError(str(exc), orig=exc)
          # Non-timeout, non-syntax operational error -> preserve legacy label/code.
          return UnknownSQLError(str(exc), code=SQL_UNKNOWN_ERROR, error_type="Database operational error", orig=exc)

      if isinstance(exc, ProgrammingError):
          return SQLSyntaxError(str(exc), orig=exc)

      if isinstance(exc, DatabaseError):
          return UnknownSQLError(str(exc), code=SQL_UNKNOWN_ERROR, error_type="Database error", orig=exc)

      return UnknownSQLError(str(exc), error_type="Unexpected error", orig=exc)
  ```

- [ ] **Step 4: Migrate `SQLSecurityError` into `errors.py` and re-export from `generate_sql.py` for back-compat.**
  Today `generate_sql.py:31-33` declares `class SQLSecurityError(ValueError)`. Replace that local class with a re-export of the new one. Remove the local definition (lines 31-33) and add the import to the `errors` module right after the existing constants import block (after line 23):

  In `openchatbi/text2sql/generate_sql.py`, delete:
  ```python
  class SQLSecurityError(ValueError):
      """Raised when generated SQL fails safety validation."""
  ```
  and add, after the `from openchatbi.graph_state import SQLGraphState` import (line 24):
  ```python
  from openchatbi.text2sql.errors import (
      RecoveryStrategy,
      SQLSecurityError,
      Text2SQLError,
      classify_sql_exception,
  )
  ```
  Note: `SQLSecurityError` is now a `Text2SQLError` (no longer a `ValueError`), but `_execute_sql` raises it directly (line 258) and `execute_sql_node` catches it (line 387) — both stay within the `generate_sql` module so the re-export keeps `from openchatbi.text2sql.generate_sql import SQLSecurityError` working for any external caller.

- [ ] **Step 5: Run the new error tests AND the existing generate_sql tests (expect PASS — no behavior change yet).**
  Run: `uv run pytest tests/test_text2sql_errors.py tests/test_text2sql_generate_sql.py -v`
  Expected: PASS for both files (`execute_sql_node` still uses its existing except chain; only the `SQLSecurityError` base class moved, and it is still raised/caught identically).

- [ ] **Step 6: Commit.**
  Run:
  ```
  git add openchatbi/text2sql/errors.py openchatbi/text2sql/generate_sql.py tests/test_text2sql_errors.py && \
  git commit -m "Add structured Text2SQL error taxonomy and classifier

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 9: Rewire `execute_sql_node` to a single `classify_sql_exception` point (preserve error_type strings; add new fields; EmptyResultError default OFF)

**Files:**
- Modify: `openchatbi/text2sql/generate_sql.py` (`_execute_sql` L246-285, `execute_sql_node` L331-416)
- Modify: `openchatbi/config_loader.py` (`Config` model — add fields needed by `_execute_sql` empty-result gate; the routing/retry fields land in Task 10)
- Modify: `tests/test_text2sql_generate_sql.py` (add new-field assertions; keep all existing `error_type` assertions intact)

- [ ] **Step 1: Write failing tests for the enriched `previous_sql_errors` entries + empty-result default-OFF.**
  Add these to `tests/test_text2sql_generate_sql.py` inside `TestText2SQLGenerateSQL`. They assert the NEW fields exist alongside the unchanged `error` / `error_type` strings, and that an empty result is still `SQL_SUCCESS` by default.

  ```python
  def test_execute_sql_node_syntax_error_enriched_fields(self, mock_llm, mock_catalog):
      """Syntax errors carry new structured fields without changing legacy strings."""
      _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
      mock_engine = mock_catalog.get_sql_engine.return_value
      mock_connection = mock_engine.connect.return_value.__enter__.return_value
      from sqlalchemy.exc import ProgrammingError

      mock_connection.execute.side_effect = ProgrammingError("", "", "Syntax error")
      state = SQLGraphState(messages=[], sql="SELECT * FRON users")

      result = execute_node(state)

      from openchatbi.constants import SQL_SYNTAX_ERROR

      entry = result["previous_sql_errors"][-1]
      # Legacy human-readable contract preserved:
      assert entry["error_type"] == "SQL syntax error"
      assert entry["error"].startswith("SQL syntax error:")
      # New structured fields:
      assert entry["error_code"] == SQL_SYNTAX_ERROR
      assert entry["error_class"] == "SQLSyntaxError"
      assert entry["recovery_strategy"] == "retry"
      assert entry["attempt"] == 1

  def test_execute_sql_node_security_error_enriched_fields(self, mock_llm, mock_catalog):
      """Security errors keep legacy strings and gain surface_to_user strategy."""
      _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
      state = SQLGraphState(messages=[], sql="SELECT * FROM users; DELETE FROM users")

      result = execute_node(state)

      from openchatbi.constants import SQL_SECURITY_ERROR

      entry = result["previous_sql_errors"][-1]
      assert entry["error_type"] == "SQL security error"
      assert entry["error_code"] == SQL_SECURITY_ERROR
      assert entry["error_class"] == "SQLSecurityError"
      assert entry["recovery_strategy"] == "surface_to_user"

  def test_execute_sql_node_attempt_increments_with_history(self, mock_llm, mock_catalog):
      """attempt counts existing previous_sql_errors + 1."""
      _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
      mock_engine = mock_catalog.get_sql_engine.return_value
      mock_connection = mock_engine.connect.return_value.__enter__.return_value
      from sqlalchemy.exc import ProgrammingError

      mock_connection.execute.side_effect = ProgrammingError("", "", "Syntax error")
      state = SQLGraphState(
          messages=[],
          sql="SELECT * FRON users",
          previous_sql_errors=[
              {"sql": "x", "error": "SQL syntax error: x", "error_type": "SQL syntax error"}
          ],
      )

      result = execute_node(state)
      assert result["previous_sql_errors"][-1]["attempt"] == 2

  def test_execute_sql_node_empty_result_default_success(self, mock_llm, mock_catalog):
      """Zero-row results stay SQL_SUCCESS when the empty-result gate is off (default)."""
      _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
      mock_engine = mock_catalog.get_sql_engine.return_value
      mock_connection = mock_engine.connect.return_value.__enter__.return_value
      mock_result = mock_connection.execute.return_value
      mock_result.fetchmany.return_value = []
      mock_result.fetchall.return_value = []

      state = SQLGraphState(messages=[], sql="SELECT * FROM users")
      result = execute_node(state)

      assert result["sql_execution_result"] == SQL_SUCCESS
  ```

- [ ] **Step 2: Run the new tests (expect failure — new fields not emitted yet).**
  Run: `uv run pytest tests/test_text2sql_generate_sql.py -v -k "enriched_fields or attempt_increments or empty_result_default"`
  Expected: FAIL — `KeyError: 'error_code'` (entries lack the new fields).

- [ ] **Step 3: Add the empty-result-gate config field to `Config`.**
  In `openchatbi/config_loader.py`, after the SQL result limit block (lines 78-80), add the field so `cls(**config)` doesn't silently drop it (pydantic `extra='ignore'`). The retry/strategy fields are added in Task 10; here we only need the empty-result switch consumed by `_execute_sql`.

  ```python
      # SQL Execution Result Limit Configuration
      enable_sql_result_limit: bool = True
      sql_result_limit: int = SQL_RESULT_LIMIT

      # Treat zero-row results as a soft failure (EmptyResultError). Default OFF:
      # empty results stay SQL_SUCCESS to preserve existing visualization-entry behavior.
      enable_empty_result_error: bool = False
  ```

- [ ] **Step 4: Collapse the except chain in `execute_sql_node` to a single `classify_sql_exception` point, emitting new fields and reusing legacy strings.**
  Replace the whole `try/except` block (lines 344-416) with the version below. Key invariants: `error_type` strings unchanged; `error` string format `"{error_type}: {str(e)}"` unchanged; new fields (`error_code`/`error_class`/`recovery_strategy`/`attempt`) added; `messages` still emitted only for the security/timeout branches that emitted them before (so the existing `result["messages"][0].content` assertions don't break); `SQL_EXECUTE_TIMEOUT` referenced as the symbol.

  ```python
      try:
          schema_info, csv_result = _execute_sql(sql_query)
          row_count = schema_info.get("row_count", 0)
          empty_result_enabled, _ = _get_empty_result_config()
          if empty_result_enabled and row_count == 0:
              previous_errors = list(state.get("previous_sql_errors", []))
              attempt = len(previous_errors) + 1
              err = EmptyResultError("Query returned no rows")
              previous_errors.append(
                  {
                      "sql": sql_query,
                      "error": f"{err.error_type}: no rows returned",
                      "error_type": err.error_type,
                      "error_code": err.code,
                      "error_class": type(err).__name__,
                      "recovery_strategy": err.recovery_strategy.value,
                      "attempt": attempt,
                  }
              )
              return {
                  "sql_execution_result": SQL_NA,
                  "previous_sql_errors": previous_errors,
              }
          if "result_limit" in schema_info:
              result_label = f"SQL Result (limited to first {schema_info['result_limit']} rows)"
          else:
              result_label = "SQL Result"
          result = f"```sql\n{sql_query}\n```\n{result_label}:\n```csv\n{csv_result}\n```"
          return {
              "sql_execution_result": SQL_SUCCESS,
              "schema_info": schema_info,
              "data": csv_result,
              "messages": [AIMessage(result)],
          }
      except Exception as e:
          err = classify_sql_exception(e)
          log(f"{err.error_type}: {str(e)}")

          previous_errors = list(state.get("previous_sql_errors", []))
          attempt = len(previous_errors) + 1
          previous_errors.append(
              {
                  "sql": sql_query,
                  "error": f"{err.error_type}: {str(e)}",
                  "error_type": err.error_type,
                  "error_code": err.code,
                  "error_class": type(err).__name__,
                  "recovery_strategy": err.recovery_strategy.value,
                  "attempt": attempt,
              }
          )

          update: dict = {
              "sql_execution_result": err.code,
              "previous_sql_errors": previous_errors,
          }
          # Branches that historically surfaced a message to the user keep doing so.
          if err.code == SQL_EXECUTE_TIMEOUT:
              error_result = (
                  f"```sql\n{sql_query}\n```\nDatabase Connection Timeout: {str(e)}\n"
                  "Please check database connectivity."
              )
              update["messages"] = [AIMessage(error_result)]
          elif err.code == SQL_SECURITY_ERROR:
              error_result = f"```sql\n{sql_query}\n```\n{err.error_type}: {str(e)}"
              update["messages"] = [AIMessage(error_result)]
          return update
  ```

  Note on the legacy timeout branch: previously a bare `TimeoutError` and an operational `timeout_or_connection` both returned `SQL_EXECUTE_TIMEOUT` with NO `previous_sql_errors` entry. Under the new single point they now also get a `previous_sql_errors` entry; no existing test asserts on the *absence* of that entry for timeouts (`test_sql_error_handling_database_error` and `test_sql_error_handling_operational_timeout_takes_priority` only assert `sql_execution_result == SQL_EXECUTE_TIMEOUT`), so this is additive and safe.

- [ ] **Step 5: Add the `_get_empty_result_config` helper next to `_get_sql_result_limit_config`.**
  Insert after `_get_sql_result_limit_config` (after line 83 in `generate_sql.py`):

  ```python
  def _get_empty_result_config() -> tuple[bool, None]:
      """Read whether zero-row results should be treated as a soft failure.

      Defaults to OFF so empty results stay SQL_SUCCESS (preserves the existing
      generate_visualization entry path). Returns a tuple for symmetry with the
      result-limit config reader.
      """
      try:
          cfg = config.get()
      except ValueError:
          return False, None
      return bool(getattr(cfg, "enable_empty_result_error", False)), None
  ```

  And add `EmptyResultError` to the `from openchatbi.text2sql.errors import (...)` block added in Task 8 Step 4:
  ```python
  from openchatbi.text2sql.errors import (
      EmptyResultError,
      RecoveryStrategy,
      SQLSecurityError,
      Text2SQLError,
      classify_sql_exception,
  )
  ```

- [ ] **Step 6: Run the full generate_sql test suite + the new error tests (expect PASS).**
  Run: `uv run pytest tests/test_text2sql_generate_sql.py tests/test_text2sql_errors.py -v`
  Expected: PASS for all — the 7+ legacy `error_type` assertions (`'SQL syntax error'`, `'SQL security error'`, `'Database operational error'`) still hold, the `result["messages"][0].content` assertions for limit/security branches still hold, and the new field/empty-result tests pass.

- [ ] **Step 7: Commit.**
  Run:
  ```
  git add openchatbi/text2sql/generate_sql.py openchatbi/config_loader.py tests/test_text2sql_generate_sql.py && \
  git commit -m "Route execute_sql_node errors through classify_sql_exception with structured fields

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 10: Strategy-driven retry routing in `_should_generate_visualization_or_retry`; config-driven `sql_max_retries`; enrich regenerate_sql_node Attempt entries

**Files:**
- Modify: `openchatbi/text2sql/sql_graph.py` (`_should_generate_visualization_or_retry` L40-58; imports L13)
- Modify: `openchatbi/config_loader.py` (`Config` — add `sql_max_retries`, `retry_on_timeout`, `retry_strategy_overrides`)
- Modify: `openchatbi/graph_state.py` (`SQLOutputState` L61-69 — add `recovery_strategy` so the field survives the subgraph output boundary)
- Modify: `openchatbi/text2sql/generate_sql.py` (`regenerate_sql_node` L418-462 — incremental: add `error_type` hint per Attempt; surface last `recovery_strategy`)
- Modify: `tests/test_text2sql_generate_sql.py` (routing + config tests)

- [ ] **Step 1: Write failing tests for strategy-driven routing + config-driven max retries + regenerate Attempt hint.**
  Add to `tests/test_text2sql_generate_sql.py`. Routing reads the last `recovery_strategy` in `previous_sql_errors`: `RETRY`/`RETRY_WITH_NEW_TABLE` → regenerate (subject to count), `SURFACE_TO_USER`/`ABORT` → end. The existing 4 routing tests (lines 442-468) must still pass — those states carry no `previous_sql_errors`, so behavior falls back to the legacy code-based check.

  ```python
  def test_routing_surface_to_user_ends(self):
      """recovery_strategy=surface_to_user ends instead of retrying."""
      from openchatbi.constants import SQL_SECURITY_ERROR

      state = SQLGraphState(
          sql_execution_result=SQL_SECURITY_ERROR,
          sql_retry_count=0,
          previous_sql_errors=[{"recovery_strategy": "surface_to_user"}],
      )
      assert _should_generate_visualization_or_retry(state) == "end"

  def test_routing_abort_ends(self):
      """recovery_strategy=abort ends (timeout case)."""
      from openchatbi.constants import SQL_EXECUTE_TIMEOUT

      state = SQLGraphState(
          sql_execution_result=SQL_EXECUTE_TIMEOUT,
          sql_retry_count=0,
          previous_sql_errors=[{"recovery_strategy": "abort"}],
      )
      assert _should_generate_visualization_or_retry(state) == "end"

  def test_routing_retry_regenerates(self):
      """recovery_strategy=retry routes to regenerate when under the retry cap."""
      from openchatbi.constants import SQL_SYNTAX_ERROR

      state = SQLGraphState(
          sql_execution_result=SQL_SYNTAX_ERROR,
          sql_retry_count=1,
          previous_sql_errors=[{"recovery_strategy": "retry"}],
      )
      assert _should_generate_visualization_or_retry(state) == "regenerate_sql"

  def test_routing_retry_respects_config_max(self):
      """sql_max_retries comes from Config, not a hardcoded 3."""
      from types import SimpleNamespace

      from openchatbi.constants import SQL_SYNTAX_ERROR

      state = SQLGraphState(
          sql_execution_result=SQL_SYNTAX_ERROR,
          sql_retry_count=1,
          previous_sql_errors=[{"recovery_strategy": "retry"}],
      )
      with patch("openchatbi.text2sql.sql_graph.config") as mock_cfg:
          mock_cfg.get.return_value = SimpleNamespace(sql_max_retries=1, retry_on_timeout=False)
          assert _should_generate_visualization_or_retry(state) == "end"

  def test_routing_no_strategy_falls_back_to_legacy(self):
      """Without recovery_strategy, legacy code-based routing still applies."""
      state = SQLGraphState(sql_execution_result="SYNTAX_ERROR", sql_retry_count=1)
      assert _should_generate_visualization_or_retry(state) == "regenerate_sql"

  def test_regenerate_attempt_includes_error_type_hint(self, mock_llm, mock_catalog):
      """Cumulative regenerate prompt annotates each Attempt with its error_type."""
      captured = {}

      def _capture(messages):
          captured["prompt"] = messages[-1].content
          return AIMessage(content="SELECT 1")

      mock_llm.invoke.side_effect = _capture
      _, _, regenerate_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

      state = SQLGraphState(
          messages=[],
          rewrite_question="Show all users",
          tables=[{"table": "users", "columns": []}],
          previous_sql_errors=[
              {
                  "sql": "SELECT * FRON users",
                  "error": "SQL syntax error: bad",
                  "error_type": "SQL syntax error",
              }
          ],
          sql_retry_count=1,
      )
      with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
          mock_retriever.invoke.return_value = []
          regenerate_node(state)

      assert "Error type: SQL syntax error" in captured["prompt"]
  ```

- [ ] **Step 2: Run the new tests (expect failure).**
  Run: `uv run pytest tests/test_text2sql_generate_sql.py -v -k "routing or regenerate_attempt_includes"`
  Expected: FAIL — `_should_generate_visualization_or_retry` ignores `recovery_strategy` / hardcodes `max_retries = 3`; regenerate prompt has no `Error type:` line.

- [ ] **Step 3: Add retry/strategy config fields to `Config`.**
  In `openchatbi/config_loader.py`, after the `enable_empty_result_error` field added in Task 9, add:

  ```python
      # SQL Retry / Recovery Configuration
      sql_max_retries: int = 3
      # When True, timeout/connection failures may be retried; default OFF keeps
      # the existing "timeout ends immediately" behavior.
      retry_on_timeout: bool = False
      # Optional per-error-class strategy overrides, e.g. {"SQLSyntaxError": "retry"}.
      # Reserved for phase-2 (e.g. enabling RETRY_WITH_NEW_TABLE); default empty = no override.
      retry_strategy_overrides: dict[str, Any] = {}
  ```

- [ ] **Step 4: Add `recovery_strategy` to `SQLOutputState`.**
  The subgraph output schema (`graph_state.py:61-69`) drops any field it doesn't declare. If the main graph is to consume the strategy (e.g. a future `SURFACE_TO_USER`→AskHuman hop), the field must survive the boundary. Add to `SQLOutputState`:

  ```python
  class SQLOutputState(MessagesState):
      """Output state schema for the SQL generation subgraph."""

      rewrite_question: str
      tables: list[dict[str, Any]]
      sql: str
      schema_info: dict[str, Any]  # Data schema analysis results
      data: str  # CSV data for display
      visualization_dsl: dict[str, Any]
      recovery_strategy: str  # Last error's recovery strategy (empty if none); see RecoveryStrategy
  ```

- [ ] **Step 5: Rewrite `_should_generate_visualization_or_retry` for strategy-driven routing + config-driven max retries.**
  In `openchatbi/text2sql/sql_graph.py`, add the imports and replace the function body (lines 40-58). `RETRY_WITH_NEW_TABLE` is treated like `RETRY` here (route to `regenerate_sql`) — the dedicated change-table edge is phase-2 and config-off (the `table_selection` edge at `sql_graph.py:106` hardwires to `generate_sql`, bypassing the cumulative-error prompt), so we do NOT add a new edge in this task.

  Add to the import block (after line 13's constants import):
  ```python
  from openchatbi.text2sql.errors import RecoveryStrategy
  ```

  Replace the function:
  ```python
  def _get_sql_retry_config() -> tuple[int, bool]:
      """Read retry settings from Config, defaulting to legacy values."""
      try:
          cfg = config.get()
      except ValueError:
          return 3, False
      max_retries = getattr(cfg, "sql_max_retries", 3)
      if not isinstance(max_retries, int) or max_retries < 0:
          max_retries = 3
      return max_retries, bool(getattr(cfg, "retry_on_timeout", False))


  def _should_generate_visualization_or_retry(state: SQLGraphState) -> str:
      """Conditional edge function to determine next action after execute_sql.

      Routing is strategy-driven: the last classified error's recovery_strategy
      decides whether to regenerate or end. Falls back to legacy code-based routing
      when no recovery_strategy is present (e.g. timeouts, untouched states).

      Args:
          state (SQLGraphState): Current state

      Returns:
          str: "generate_visualization" on success, "regenerate_sql" to retry, "end" otherwise.
      """
      execution_result = state.get("sql_execution_result", "")
      retry_count = state.get("sql_retry_count", 0)
      max_retries, retry_on_timeout = _get_sql_retry_config()

      if execution_result == SQL_SUCCESS:
          return "generate_visualization"

      previous_errors = state.get("previous_sql_errors", [])
      strategy = previous_errors[-1].get("recovery_strategy") if previous_errors else None

      if strategy is not None:
          if strategy in (RecoveryStrategy.SURFACE_TO_USER.value, RecoveryStrategy.ABORT.value):
              return "end"
          if strategy in (RecoveryStrategy.RETRY.value, RecoveryStrategy.RETRY_WITH_NEW_TABLE.value):
              return "regenerate_sql" if retry_count < max_retries else "end"
          return "end"

      # Legacy fallback: no structured strategy recorded.
      if retry_count < max_retries and (
          execution_result != SQL_EXECUTE_TIMEOUT or retry_on_timeout
      ):
          return "regenerate_sql"
      return "end"
  ```

- [ ] **Step 6: Enrich `regenerate_sql_node` Attempt entries with an `error_type` hint and surface the last `recovery_strategy`.**
  This is incremental — the cumulative loop already exists at `generate_sql.py:440-444`. Modify the loop to append the per-Attempt error type, and add `recovery_strategy` to both return dicts so it reaches `SQLOutputState`. Replace lines 440-444:

  ```python
          if previous_errors:
              user_prompt += "\n\nPrevious attempts failed with errors:"
              for i, error_info in enumerate(previous_errors, 1):
                  error_type_hint = error_info.get("error_type", "")
                  hint_line = f"\nError type: {error_type_hint}" if error_type_hint else ""
                  user_prompt += (
                      f"\n\nAttempt {i}:\nSQL: {error_info['sql']}"
                      f"{hint_line}\nError: {error_info['error']}"
                  )
              user_prompt += "\n\nPlease analyze the errors above and generate a corrected SQL query."
  ```

  And surface `recovery_strategy` on the returns. Replace the empty-response return block (lines 455-460):
  ```python
          last_strategy = previous_errors[-1].get("recovery_strategy", "") if previous_errors else ""
          if not sql_query:
              log(f"Generated SQL query is empty. LLM output: {response.content}")
              error_result = f"Failed to regenerate valid SQL after {retry_count} attempts."
              return {
                  "messages": [AIMessage(error_result)],
                  "sql": "",
                  "sql_retry_count": retry_count,
                  "sql_execution_result": SQL_NA,
                  "recovery_strategy": last_strategy,
              }

          return {
              "sql": sql_query,
              "sql_retry_count": retry_count,
              "sql_execution_result": "",
              "recovery_strategy": last_strategy,
          }
  ```
  (`previous_errors` is already bound at line 429; `last_strategy` is computed once before both returns.)

- [ ] **Step 7: Run routing, regenerate, and the full generate_sql suite (expect PASS).**
  Run: `uv run pytest tests/test_text2sql_generate_sql.py -v`
  Expected: PASS — the 4 legacy routing tests (lines 442-468) still pass via the legacy fallback (no `recovery_strategy` in their states; `config.get()` raises `ValueError` in unit context → defaults `max_retries=3, retry_on_timeout=False`), the new strategy/config tests pass, and `test_regenerate_sql_node_success` still passes (entry lacks `recovery_strategy` → empty string, hint line omitted when `error_type` absent).

- [ ] **Step 8: Commit.**
  Run:
  ```
  git add openchatbi/text2sql/sql_graph.py openchatbi/config_loader.py openchatbi/graph_state.py openchatbi/text2sql/generate_sql.py tests/test_text2sql_generate_sql.py && \
  git commit -m "Strategy-driven SQL retry routing with config-driven max retries

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## HITL(Tasks 11-12)

### Task 11: Confidence scoring node + confidence gate (HITL post-exec review)

**Files:**
- Modify `openchatbi/config_loader.py` (add `enable_confidence_gate`, `sql_confidence_threshold`, `confidence_gate_mode` fields to `Config`, after `sql_result_limit` block ~L80)
- Modify `openchatbi/graph_state.py` (add `sql_confidence`/`confidence_reasons`/`human_sql_decision` to `SQLGraphState` L31-46 AND `SQLOutputState` L61-69)
- Modify `openchatbi/streaming.py` (add `"confidence"` to `SQL_SUBGRAPH_NODES` L29-38; add `score_sql`/`confidence_gate` handling + `StreamStep(kind="confidence")` emit in `_process_updates` ~L304)
- Modify `openchatbi/text2sql/generate_sql.py` (add `score_sql_node` + `confidence_gate_node` factory funcs; extend `create_sql_nodes` return tuple, L136-505)
- Modify `openchatbi/text2sql/sql_graph.py` (add `route_after_confidence`; wire `score_sql`/`confidence_gate` nodes + reroute `execute_sql` success edge; `_should_generate_visualization_or_retry` L40-58, `build_sql_graph` L61-156)
- Create `tests/test_text2sql_confidence_gate.py`

- [ ] **Step 1: Write failing test for Config confidence-gate flags (default OFF).**
  Add to a new test file `tests/test_text2sql_confidence_gate.py`:
  ```python
  """Tests for HITL confidence scoring node and confidence gate."""

  from types import SimpleNamespace
  from unittest.mock import Mock, patch

  import pytest
  from langchain_core.messages import AIMessage, HumanMessage
  from langgraph.checkpoint.memory import MemorySaver
  from langgraph.types import Command

  from openchatbi.config_loader import Config
  from openchatbi.constants import SQL_SUCCESS
  from openchatbi.graph_state import SQLGraphState, SQLOutputState
  from openchatbi.text2sql.confidence import ConfidenceResult


  class TestConfidenceGateConfig:
      def test_confidence_flags_default_off(self):
          cfg = Config(default_llm=Mock(), data_warehouse_config={"uri": "sqlite:///:memory:"})
          assert cfg.enable_confidence_gate is False
          assert cfg.sql_confidence_threshold == 0.7
          assert cfg.confidence_gate_mode == "post_exec"

      def test_confidence_flags_from_dict(self):
          cfg = Config.from_dict(
              {
                  "default_llm": Mock(),
                  "data_warehouse_config": {"uri": "sqlite:///:memory:"},
                  "enable_confidence_gate": True,
                  "sql_confidence_threshold": 0.5,
                  "confidence_gate_mode": "pre_exec",
              }
          )
          assert cfg.enable_confidence_gate is True
          assert cfg.sql_confidence_threshold == 0.5
          assert cfg.confidence_gate_mode == "pre_exec"
  ```
- [ ] **Step 2: Run the test — expect FAIL (fields not declared, pydantic drops them).**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceGateConfig -v`
  Expected: FAIL — `AttributeError: 'Config' object has no attribute 'enable_confidence_gate'` (extra='ignore' silently drops undeclared keys).

- [ ] **Step 3: Declare the confidence-gate fields on `Config`.**
  In `openchatbi/config_loader.py`, after the SQL result-limit block (L78-80):
  ```python
      # SQL Execution Result Limit Configuration
      enable_sql_result_limit: bool = True
      sql_result_limit: int = SQL_RESULT_LIMIT

      # HITL Confidence Gate Configuration (default OFF for zero-regression)
      enable_confidence_gate: bool = False
      sql_confidence_threshold: float = 0.7
      confidence_gate_mode: str = "post_exec"  # Options: "post_exec" (default), "pre_exec" (phase-2)
      confidence_evaluator_mode: str = "simple"
  ```
- [ ] **Step 4: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceGateConfig -v`
  Expected: PASS.

- [ ] **Step 5: Commit.**
  Run: `git add openchatbi/config_loader.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): add confidence-gate Config flags (default off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 6: Write failing test for the new state fields surviving the subgraph boundary.**
  Append to `tests/test_text2sql_confidence_gate.py`:
  ```python
  class TestConfidenceStateFields:
      def test_sqlgraphstate_accepts_confidence_fields(self):
          state = SQLGraphState(
              messages=[HumanMessage(content="q")],
              sql="SELECT 1",
              sql_confidence=0.42,
              confidence_reasons=["WHERE clause missing filter"],
              human_sql_decision="approve",
          )
          assert state["sql_confidence"] == 0.42
          assert state["confidence_reasons"] == ["WHERE clause missing filter"]
          assert state["human_sql_decision"] == "approve"

      def test_sqloutputstate_exposes_confidence_fields(self):
          # SQLOutputState is the subgraph output schema; fields absent here are
          # filtered out at the subgraph boundary and never reach the parent graph.
          assert "sql_confidence" in SQLOutputState.__annotations__
          assert "confidence_reasons" in SQLOutputState.__annotations__
          assert "human_sql_decision" in SQLOutputState.__annotations__
  ```
- [ ] **Step 7: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceStateFields -v`
  Expected: FAIL — `KeyError: 'sql_confidence'` and `assert 'sql_confidence' in SQLOutputState.__annotations__` is False.

- [ ] **Step 8: Add the fields to BOTH `SQLGraphState` and `SQLOutputState`.**
  In `openchatbi/graph_state.py`, extend `SQLGraphState` (after `visualization_dsl` L46):
  ```python
      previous_sql_errors: list[dict[str, Any]]
      visualization_dsl: dict[str, Any]
      sql_confidence: float
      confidence_reasons: list[str]
      human_sql_decision: str
  ```
  And extend `SQLOutputState` (after `visualization_dsl` L69):
  ```python
      data: str  # CSV data for display
      visualization_dsl: dict[str, Any]
      sql_confidence: float
      confidence_reasons: list[str]
      human_sql_decision: str
  ```
- [ ] **Step 9: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceStateFields -v`
  Expected: PASS.

- [ ] **Step 10: Commit.**
  Run: `git add openchatbi/graph_state.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): add confidence/decision fields to SQL graph + output state

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 11: Write failing test for `score_sql_node` calling S2 `SimpleSQLEvaluator`.**
  Append to `tests/test_text2sql_confidence_gate.py`:
  ```python
  class TestScoreSqlNode:
      def _nodes(self, mock_llm, mock_catalog):
          from openchatbi.text2sql.generate_sql import create_sql_nodes

          return create_sql_nodes(mock_llm, mock_catalog, "presto")

      @pytest.fixture
      def mock_llm(self):
          llm = Mock()
          llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
          return llm

      @pytest.fixture
      def mock_catalog(self):
          return Mock()

      def test_score_sql_node_returns_confidence(self, mock_llm, mock_catalog):
          nodes = self._nodes(mock_llm, mock_catalog)
          # create_sql_nodes now returns 6 callables (was 4): + score_sql, confidence_gate
          score_sql_node = nodes[4]
          fake_result = ConfidenceResult(
              score=0.35, reasons=["WHERE missing"], checks={"where": False}
          )
          with patch(
              "openchatbi.text2sql.generate_sql.SimpleSQLEvaluator"
          ) as MockEval:
              MockEval.return_value.evaluate.return_value = fake_result
              state = SQLGraphState(
                  messages=[],
                  rewrite_question="how many users",
                  sql="SELECT * FROM users",
                  schema_info={"columns": ["id"]},
                  data="id\n1\n",
                  sql_execution_result=SQL_SUCCESS,
              )
              out = score_sql_node(state)
          assert out["sql_confidence"] == 0.35
          assert out["confidence_reasons"] == ["WHERE missing"]

      def test_score_sql_node_skips_on_failed_sql(self, mock_llm, mock_catalog):
          nodes = self._nodes(mock_llm, mock_catalog)
          score_sql_node = nodes[4]
          state = SQLGraphState(
              messages=[], rewrite_question="q", sql="SELECT 1", sql_execution_result="SQL_SYNTAX_ERROR"
          )
          out = score_sql_node(state)
          # No confidence computed for non-success executions.
          assert out == {}
  ```
- [ ] **Step 12: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestScoreSqlNode -v`
  Expected: FAIL — `create_sql_nodes` returns a 4-tuple so `nodes[4]` raises `IndexError`.

- [ ] **Step 13: Implement `score_sql_node` + `confidence_gate_node` and extend `create_sql_nodes`.**
  In `openchatbi/text2sql/generate_sql.py`, add the import near the top (after L27):
  ```python
  from openchatbi.text2sql.confidence import SimpleSQLEvaluator
  from langgraph.types import interrupt
  ```
  Inside `create_sql_nodes`, before the final `return` (L505), add:
  ```python
      def score_sql_node(state: SQLGraphState) -> dict:
          """Score the executed SQL with the S2 confidence evaluator.

          Only runs after a successful execution (post_exec mode); other
          execution results are passed through unscored.
          """
          if state.get("sql_execution_result", "") != SQL_SUCCESS:
              return {}
          sql_query = state.get("sql", "").strip()
          if not sql_query:
              return {}
          question = state.get("rewrite_question", "")
          schema_info = state.get("schema_info", {})
          data_sample = state.get("data", "")
          try:
              evaluator = SimpleSQLEvaluator(llm)
              result = evaluator.evaluate(question, sql_query, schema_info, data_sample)
          except Exception as e:  # never block the answer on evaluator failure
              log(f"Confidence evaluation failed: {str(e)}")
              return {}
          log(f"SQL confidence={result.score:.2f} reasons={result.reasons}")
          return {"sql_confidence": result.score, "confidence_reasons": list(result.reasons)}

      def confidence_gate_node(state: SQLGraphState) -> dict:
          """Interrupt for human review when confidence is below threshold.

          Reuses the ask_human interrupt channel (buttons approve/reject/edit).
          Returns the human decision (and edited SQL on 'edit').
          """
          try:
              cfg = config.get()
              enabled = bool(getattr(cfg, "enable_confidence_gate", False))
              threshold = float(getattr(cfg, "sql_confidence_threshold", 0.7))
          except ValueError:
              enabled, threshold = False, 0.7
          score = state.get("sql_confidence", 1.0)
          if not enabled or score is None or score >= threshold:
              return {"human_sql_decision": "approve"}
          reasons = state.get("confidence_reasons", [])
          feedback = interrupt(
              {
                  "text": f"Low-confidence SQL ({score:.2f}). Reasons: {'; '.join(reasons) or 'n/a'}. Approve?",
                  "buttons": ["approve", "reject", "edit"],
                  "sql": state.get("sql", ""),
              }
          )
          decision = feedback if isinstance(feedback, str) else (feedback or {}).get("decision", "approve")
          if decision == "edit":
              edited = feedback.get("sql") if isinstance(feedback, dict) else None
              if edited:
                  return {"human_sql_decision": "edit", "sql": edited}
          return {"human_sql_decision": decision}
  ```
  Change the final return to a 6-tuple:
  ```python
      return (
          generate_sql_node,
          execute_sql_node,
          regenerate_sql_node,
          generate_visualization_node,
          score_sql_node,
          confidence_gate_node,
      )
  ```
  Update the signature docstring return note (L147-148) to read `tuple: Six node functions (generate, execute, regenerate, visualization, score_sql, confidence_gate)`.

- [ ] **Step 14: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestScoreSqlNode -v`
  Expected: PASS.

- [ ] **Step 15: Commit.**
  Run: `git add openchatbi/text2sql/generate_sql.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): add score_sql_node + confidence_gate_node to create_sql_nodes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 16: Write failing test for `route_after_confidence` routing.**
  Append to `tests/test_text2sql_confidence_gate.py`:
  ```python
  class TestRouteAfterConfidence:
      def test_route_approve_goes_to_visualization(self):
          from openchatbi.text2sql.sql_graph import route_after_confidence

          assert route_after_confidence({"human_sql_decision": "approve"}) == "generate_visualization"

      def test_route_reject_goes_to_regenerate(self):
          from openchatbi.text2sql.sql_graph import route_after_confidence

          assert route_after_confidence({"human_sql_decision": "reject"}) == "regenerate_sql"

      def test_route_edit_goes_to_execute(self):
          from openchatbi.text2sql.sql_graph import route_after_confidence

          # An edited SQL must be re-executed before visualization.
          assert route_after_confidence({"human_sql_decision": "edit"}) == "execute_sql"

      def test_route_default_when_no_decision(self):
          from openchatbi.text2sql.sql_graph import route_after_confidence

          assert route_after_confidence({}) == "generate_visualization"
  ```
- [ ] **Step 17: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestRouteAfterConfidence -v`
  Expected: FAIL — `ImportError: cannot import name 'route_after_confidence' from 'openchatbi.text2sql.sql_graph'`.

- [ ] **Step 18: Implement `route_after_confidence` and wire the score/gate nodes into the graph.**
  In `openchatbi/text2sql/sql_graph.py`, add after `_should_generate_visualization_or_retry` (L58):
  ```python
  def route_after_confidence(state: SQLGraphState) -> str:
      """Route after the confidence gate based on the human decision.

      approve -> visualization; reject -> regenerate; edit -> re-execute the
      user-edited SQL. Defaults to visualization when no decision is present
      (gate disabled or score above threshold).
      """
      decision = state.get("human_sql_decision", "approve")
      if decision == "reject":
          return "regenerate_sql"
      if decision == "edit":
          return "execute_sql"
      return "generate_visualization"
  ```
  In `build_sql_graph`, unpack the new 6-tuple (L82-87):
  ```python
      (
          generate_sql_node,
          execute_sql_node,
          regenerate_sql_node,
          generate_visualization_node,
          score_sql_node,
          confidence_gate_node,
      ) = create_sql_nodes(
          get_text2sql_llm(llm_provider),
          catalog,
          dialect=config.get().dialect,
          visualization_mode=config.get().visualization_mode,
      )
  ```
  Register the nodes (after L100):
  ```python
      graph.add_node("generate_visualization", generate_visualization_node)
      graph.add_node("score_sql", score_sql_node)
      graph.add_node("confidence_gate", confidence_gate_node)
  ```
  Reroute the `execute_sql` success branch to `score_sql` instead of `generate_visualization` (replace the conditional-edge mapping at L142-150):
  ```python
      graph.add_conditional_edges(
          "execute_sql",
          _should_generate_visualization_or_retry,
          {
              "generate_visualization": "score_sql",
              "regenerate_sql": "regenerate_sql",
              "end": END,
          },
      )
      graph.add_edge("score_sql", "confidence_gate")
      graph.add_conditional_edges(
          "confidence_gate",
          route_after_confidence,
          {
              "generate_visualization": "generate_visualization",
              "regenerate_sql": "regenerate_sql",
              "execute_sql": "execute_sql",
          },
      )
  ```
  Add the `SQLGraphState` import for the new function (it is already imported at L14; no change needed).

- [ ] **Step 19: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestRouteAfterConfidence -v`
  Expected: PASS.

- [ ] **Step 20: Commit.**
  Run: `git add openchatbi/text2sql/sql_graph.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): wire score_sql/confidence_gate nodes + route_after_confidence (post_exec)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 21: Write failing test for interrupt propagation through the text2sql StructuredTool boundary + resume.**
  Append to `tests/test_text2sql_confidence_gate.py`:
  ```python
  class TestInterruptThroughToolBoundary:
      """Verify the confidence_gate interrupt survives the get_sql_tools
      StructuredTool boundary (GraphInterrupt re-raised in agent_graph) and
      that Command(resume=...) routes back to the correct node."""

      def _build_gated_graph(self):
          from langgraph.graph import START, END, StateGraph
          from openchatbi.graph_state import InputState, SQLGraphState, SQLOutputState
          from openchatbi.text2sql.sql_graph import route_after_confidence
          from openchatbi.text2sql.generate_sql import create_sql_nodes

          llm = Mock()
          llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
          catalog = Mock()
          nodes = create_sql_nodes(llm, catalog, "presto")
          confidence_gate_node = nodes[5]

          def execute_stub(state):
              return {"sql_execution_result": SQL_SUCCESS, "data": "id\n1\n", "schema_info": {}}

          def score_stub(state):
              return {"sql_confidence": 0.30, "confidence_reasons": ["WHERE missing"]}

          def viz_stub(state):
              return {"visualization_dsl": {"chart_type": "bar"}}

          g = StateGraph(SQLGraphState, input_schema=InputState, output_schema=SQLOutputState)
          g.add_node("execute_sql", execute_stub)
          g.add_node("score_sql", score_stub)
          g.add_node("confidence_gate", confidence_gate_node)
          g.add_node("generate_visualization", viz_stub)
          g.add_node("regenerate_sql", lambda s: {"sql": "SELECT 2"})
          g.add_edge(START, "execute_sql")
          g.add_edge("execute_sql", "score_sql")
          g.add_edge("score_sql", "confidence_gate")
          g.add_conditional_edges(
              "confidence_gate",
              route_after_confidence,
              {
                  "generate_visualization": "generate_visualization",
                  "regenerate_sql": "regenerate_sql",
                  "execute_sql": "execute_sql",
              },
          )
          g.add_edge("generate_visualization", END)
          g.add_edge("regenerate_sql", END)
          return g.compile(checkpointer=MemorySaver())

      def test_low_confidence_interrupts_then_resume_approve(self):
          from openchatbi.agent_graph import get_sql_tools
          from langgraph.errors import GraphInterrupt

          with patch("openchatbi.config_loader.ConfigLoader") as _:
              graph = self._build_gated_graph()
          cfg = Config(
              default_llm=Mock(),
              data_warehouse_config={"uri": "sqlite:///:memory:"},
              enable_confidence_gate=True,
              sql_confidence_threshold=0.7,
          )
          with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg):
              run_cfg = {"configurable": {"thread_id": "t-1"}}
              tool = get_sql_tools(graph, sync_mode=True)
              # The interrupt fires inside the subgraph and the text2sql tool
              # re-raises GraphInterrupt (agent_graph.py:160-162).
              with pytest.raises(GraphInterrupt):
                  tool.invoke(
                      {"reasoning": "r", "context": "how many users"}, config=run_cfg
                  )
              # Resume with the human approval -> graph completes at visualization.
              final = graph.invoke(Command(resume="approve"), config=run_cfg)
              assert final["human_sql_decision"] == "approve"
              assert final["visualization_dsl"]["chart_type"] == "bar"
  ```
- [ ] **Step 22: Run the test — expect FAIL initially, then confirm.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestInterruptThroughToolBoundary -v`
  Expected: FAIL first run if `get_sql_tools` does not pass `config` through to `sql_graph.invoke` (the interrupt's checkpoint thread is lost). This pins the requirement that the text2sql tool must forward `config`.

- [ ] **Step 23: Thread `config` through the text2sql tool so the interrupt checkpoints correctly.**
  In `openchatbi/agent_graph.py`, update `call_sql_graph_sync` (L153-166) to accept and forward a runnable config:
  ```python
      def call_sql_graph_sync(
          reasoning: str,
          context: str | dict[str, Any] | list[dict[str, Any]] | list[str],
          config: RunnableConfig | None = None,
      ) -> str:
          """Sync node function for Text2SQL tool"""
          normalized_context = _normalize_text2sql_context(context)
          log(f"Call SQL graph (sync) with reasoning: {reasoning}, context: {normalized_context}")
          try:
              sql_graph_response = sql_graph.invoke({"messages": normalized_context}, config=config)
              return _format_sql_response(sql_graph_response)
          except GraphInterrupt as e:
              log(f"Sql graph interrupted:\n{repr(e)}")
              raise e
          except Exception as e:
              log(f"Run sql graph error:\n{repr(e)}")
              traceback.print_exc()
          return "Error occurred when calling Text2SQL tool."
  ```
  Apply the same `config: RunnableConfig | None = None` parameter and `config=config` forwarding to `call_sql_graph_async` (L168-183) using `await sql_graph.ainvoke({"messages": normalized_context}, config=config)`. Add `from langchain_core.runnables import RunnableConfig` to the imports if not already present. LangChain injects the runnable `config` into a tool function that declares such a parameter, so `StructuredTool.from_function` needs no further change.

- [ ] **Step 24: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestInterruptThroughToolBoundary -v`
  Expected: PASS — the tool re-raises `GraphInterrupt`, and `Command(resume="approve")` on the same `thread_id` routes through `route_after_confidence` to `generate_visualization`.

- [ ] **Step 25: Commit.**
  Run: `git add openchatbi/agent_graph.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): forward config through text2sql tool so confidence interrupt resumes correctly

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 26: Write failing test for surfacing confidence as `StreamStep(kind='confidence')`.**
  Append to `tests/test_text2sql_confidence_gate.py`:
  ```python
  class TestConfidenceStreaming:
      def test_score_sql_emits_confidence_step(self):
          from openchatbi.streaming import AgentStreamProcessor, StreamStep

          proc = AgentStreamProcessor()
          events = proc.process(
              namespace=(),
              event_type="updates",
              event_value={"score_sql": {"sql_confidence": 0.42, "confidence_reasons": ["WHERE missing"]}},
          )
          conf = [e for e in events if isinstance(e, StreamStep) and e.kind == "confidence"]
          assert len(conf) == 1
          assert conf[0].data["sql_confidence"] == 0.42
          assert "0.42" in conf[0].text

      def test_score_sql_node_in_subgraph_nodes(self):
          from openchatbi.streaming import SQL_SUBGRAPH_NODES

          assert "score_sql" in SQL_SUBGRAPH_NODES
          assert "confidence_gate" in SQL_SUBGRAPH_NODES
  ```
- [ ] **Step 27: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceStreaming -v`
  Expected: FAIL — no `confidence` kind emitted; `"score_sql"` not in `SQL_SUBGRAPH_NODES`.

- [ ] **Step 28: Add `score_sql`/`confidence_gate` to subgraph nodes and emit the confidence step.**
  In `openchatbi/streaming.py`, extend `SQL_SUBGRAPH_NODES` (L29-38):
  ```python
  SQL_SUBGRAPH_NODES = {
      "search_knowledge",
      "ask_human",
      "information_extraction",
      "table_selection",
      "generate_sql",
      "execute_sql",
      "regenerate_sql",
      "generate_visualization",
      "score_sql",
      "confidence_gate",
  }
  ```
  In `_process_updates`, add a handler before the `elif node_name == "use_tool":` branch (L287):
  ```python
          elif node_name == "score_sql":
              score = node_output.get("sql_confidence")
              if score is not None:
                  reasons = node_output.get("confidence_reasons", [])
                  reason_txt = f" — {'; '.join(reasons)}" if reasons else ""
                  desc = f"🎯 SQL confidence: {score:.2f}{reason_txt}"
                  kind = "confidence"
                  data = {"sql_confidence": score, "confidence_reasons": reasons}
  ```
- [ ] **Step 29: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py::TestConfidenceStreaming -v`
  Expected: PASS.

- [ ] **Step 30: Run the full new test module + the existing generate_sql/streaming suites to confirm zero regression.**
  Run: `uv run pytest tests/test_text2sql_confidence_gate.py tests/test_text2sql_generate_sql.py tests/test_streaming.py -v`
  Expected: PASS — the default `confidence_gate_mode='post_exec'` with `enable_confidence_gate=False` leaves prior behavior intact (gate returns `approve` immediately, no interrupt).

- [ ] **Step 31: Commit.**
  Run: `git add openchatbi/streaming.py tests/test_text2sql_confidence_gate.py && git commit -m "feat(hitl): surface SQL confidence via StreamStep(kind='confidence')

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

### Task 12: Golden-SQL capture on approve + sql_examples KB branch

**Files:**
- Modify `openchatbi/config_loader.py` (add `enable_golden_sql`, `golden_sql_namespace` fields to `Config`, after the confidence-gate block from Task 11)
- Modify `openchatbi/catalog/catalog_store.py` (add abstract `append_sql_example`, after `save_table_sql_examples` L145-158)
- Modify `openchatbi/catalog/store/file_system.py` (implement `append_sql_example` with append+dedup, after `save_table_sql_examples` L731-763)
- Modify `openchatbi/text2sql/generate_sql.py` (golden capture in `confidence_gate_node` approve branch — depends on Task 11)
- Modify `openchatbi/tool/search_knowledge.py` (add `sql_examples` KB branch + fix `business` never-branched bug, L36-39)
- Create `tests/test_golden_sql_capture.py`

- [ ] **Step 1: Write failing test for golden-SQL Config flags (default OFF).**
  Create `tests/test_golden_sql_capture.py`:
  ```python
  """Tests for Golden-SQL capture flow and sql_examples KB branch."""

  import threading
  from types import SimpleNamespace
  from unittest.mock import Mock, patch

  import pytest

  from openchatbi.config_loader import Config


  class TestGoldenSqlConfig:
      def test_golden_flags_default_off(self):
          cfg = Config(default_llm=Mock(), data_warehouse_config={"uri": "sqlite:///:memory:"})
          assert cfg.enable_golden_sql is False
          assert cfg.golden_sql_namespace == "global"

      def test_golden_flags_from_dict(self):
          cfg = Config.from_dict(
              {
                  "default_llm": Mock(),
                  "data_warehouse_config": {"uri": "sqlite:///:memory:"},
                  "enable_golden_sql": True,
                  "golden_sql_namespace": "team_a",
              }
          )
          assert cfg.enable_golden_sql is True
          assert cfg.golden_sql_namespace == "team_a"
  ```
- [ ] **Step 2: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestGoldenSqlConfig -v`
  Expected: FAIL — `AttributeError: 'Config' object has no attribute 'enable_golden_sql'`.

- [ ] **Step 3: Declare the golden-SQL fields on `Config`.**
  In `openchatbi/config_loader.py`, after the HITL confidence-gate block added in Task 11:
  ```python
      # HITL Golden-SQL Capture Configuration (default OFF for zero-regression)
      enable_golden_sql: bool = False
      golden_sql_namespace: str = "global"
  ```
- [ ] **Step 4: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestGoldenSqlConfig -v`
  Expected: PASS.

- [ ] **Step 5: Commit.**
  Run: `git add openchatbi/config_loader.py tests/test_golden_sql_capture.py && git commit -m "feat(hitl): add golden-SQL Config flags (default off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 6: Write failing test for `append_sql_example` (append+dedup, NOT overwrite).**
  Append to `tests/test_golden_sql_capture.py`:
  ```python
  class TestAppendSqlExample:
      def test_append_adds_new_example_without_overwriting(self, mock_catalog_store):
          # save_table_sql_examples overwrites; append_sql_example must keep prior ones.
          mock_catalog_store.save_table_sql_examples(
              "test.test_table", [{"question": "count rows", "answer": "SELECT COUNT(*) FROM test_table"}]
          )
          ok = mock_catalog_store.append_sql_example(
              "how many names", "SELECT COUNT(name) FROM test_table", ["test.test_table"], source="golden"
          )
          assert ok is True
          examples = mock_catalog_store.get_sql_examples()
          questions = {q for q, _sql, _t in examples}
          assert "count rows" in questions
          assert "how many names" in questions

      def test_append_dedups_identical_question(self, mock_catalog_store):
          mock_catalog_store.append_sql_example(
              "dup q", "SELECT 1 FROM test_table", ["test.test_table"], source="golden"
          )
          mock_catalog_store.append_sql_example(
              "dup q", "SELECT 1 FROM test_table", ["test.test_table"], source="golden"
          )
          examples = mock_catalog_store.get_sql_examples()
          dup = [q for q, _sql, _t in examples if q == "dup q"]
          assert len(dup) == 1
  ```
  (`mock_catalog_store` is the existing `tests/conftest.py` fixture, a `FileSystemCatalogStore`.)
- [ ] **Step 7: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestAppendSqlExample -v`
  Expected: FAIL — `AttributeError: 'FileSystemCatalogStore' object has no attribute 'append_sql_example'`.

- [ ] **Step 8: Add the abstract method to `CatalogStore`.**
  In `openchatbi/catalog/catalog_store.py`, after `save_table_sql_examples` (L145-158):
  ```python
      @abstractmethod
      def append_sql_example(
          self,
          question: str,
          sql: str,
          tables: list[str],
          source: str = "golden",
          database: str | None = None,
      ) -> bool:
          """Append a single Q->SQL example, de-duplicating on the question.

          Unlike ``save_table_sql_examples`` (which overwrites the per-table
          example block), this preserves existing examples and only adds the
          new one when its question is not already present.

          Args:
              question (str): The natural-language question.
              sql (str): The validated SQL answer.
              tables (list[str]): Full table names referenced by the SQL.
              source (str): Provenance marker ('golden' for human-approved).
              database (Optional[str]): Database name override.

          Returns:
              bool: Whether the append succeeded.
          """
          pass
  ```
- [ ] **Step 9: Implement `append_sql_example` in `FileSystemCatalogStore`.**
  In `openchatbi/catalog/store/file_system.py`, after `save_table_sql_examples` (L763):
  ```python
      def append_sql_example(
          self,
          question: str,
          sql: str,
          tables: list[str],
          source: str = "golden",
          database: str | None = None,
      ) -> bool:
          self._validate_sql_examples([{"question": question, "answer": sql}])
          try:
              target_table = tables[0] if tables else ""
              full_table_name, db_name, table_name = split_db_table_name(target_table, database)

              sql_examples = self._load_yaml_file(self.sql_example_file)
              if db_name not in sql_examples:
                  sql_examples[db_name] = {}
              existing = sql_examples[db_name].get(table_name, "")

              # De-dup on the question text (the "Q: ..." line) — append only if new.
              if f"Q: {question}\n" in existing:
                  logger.info(f"Golden SQL example already present for table {full_table_name}; skipping append.")
                  return True

              new_block = f"Q: {question}\nA: {sql}\n"
              sql_examples[db_name][table_name] = (existing + "\n\n" + new_block).strip() if existing else new_block

              success = self._save_yaml_file(self.sql_example_file, sql_examples)
              if success:
                  logger.info(f"Appended {source} SQL example for table {full_table_name}")
                  self._sql_example_cache = sql_examples
              return success
          except Exception as e:
              logger.error(f"Unexpected error when appending SQL example: {e}")
              logger.error(traceback.format_stack())
              return False
  ```
- [ ] **Step 10: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestAppendSqlExample -v`
  Expected: PASS.

- [ ] **Step 11: Commit.**
  Run: `git add openchatbi/catalog/catalog_store.py openchatbi/catalog/store/file_system.py tests/test_golden_sql_capture.py && git commit -m "feat(hitl): add append_sql_example (append+dedup, not overwrite) to catalog store

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 12: Write failing test for golden capture in the `confidence_gate_node` approve branch (dual-write).**
  Append to `tests/test_golden_sql_capture.py`:
  ```python
  class TestGoldenCaptureOnApprove:
      def _gate_node(self, mock_catalog_store):
          from openchatbi.text2sql.generate_sql import create_sql_nodes

          llm = Mock()
          nodes = create_sql_nodes(llm, mock_catalog_store, "presto")
          return nodes[5]  # confidence_gate_node

      def test_approve_dual_writes_golden_sql(self, mock_catalog_store):
          from openchatbi.config_loader import Config
          from openchatbi.constants import SQL_SUCCESS
          from openchatbi.graph_state import SQLGraphState

          cfg = Config(
              default_llm=Mock(),
              data_warehouse_config={"uri": "sqlite:///:memory:"},
              catalog_store=mock_catalog_store,
              enable_confidence_gate=True,
              sql_confidence_threshold=0.7,
              enable_golden_sql=True,
              golden_sql_namespace="global",
          )
          learned_store = Mock()
          gate = self._gate_node(mock_catalog_store)
          state = SQLGraphState(
              messages=[],
              rewrite_question="how many names",
              sql="SELECT COUNT(name) FROM test_table",
              tables=[{"table": "test.test_table"}],
              sql_confidence=0.95,  # >= threshold -> auto-approve, no interrupt
              sql_execution_result=SQL_SUCCESS,
          )
          with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg), patch(
              "openchatbi.text2sql.generate_sql.get_learned_sql_store", return_value=learned_store
          ):
              out = gate(state)
          assert out["human_sql_decision"] == "approve"
          # Vector-store write (S3) ...
          learned_store.add_golden_sql.assert_called_once()
          # ... and durable YAML write (catalog) both happened.
          examples = mock_catalog_store.get_sql_examples()
          assert any(q == "how many names" for q, _sql, _t in examples)

      def test_approve_skips_capture_when_golden_disabled(self, mock_catalog_store):
          from openchatbi.config_loader import Config
          from openchatbi.constants import SQL_SUCCESS
          from openchatbi.graph_state import SQLGraphState

          cfg = Config(
              default_llm=Mock(),
              data_warehouse_config={"uri": "sqlite:///:memory:"},
              catalog_store=mock_catalog_store,
              enable_confidence_gate=True,
              sql_confidence_threshold=0.7,
              enable_golden_sql=False,
          )
          learned_store = Mock()
          gate = self._gate_node(mock_catalog_store)
          state = SQLGraphState(
              messages=[],
              rewrite_question="q",
              sql="SELECT 1 FROM test_table",
              tables=[{"table": "test.test_table"}],
              sql_confidence=0.95,
              sql_execution_result=SQL_SUCCESS,
          )
          with patch("openchatbi.text2sql.generate_sql.config.get", return_value=cfg), patch(
              "openchatbi.text2sql.generate_sql.get_learned_sql_store", return_value=learned_store
          ):
              gate(state)
          learned_store.add_golden_sql.assert_not_called()
  ```
- [ ] **Step 13: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestGoldenCaptureOnApprove -v`
  Expected: FAIL — `confidence_gate_node` does not capture golden SQL; `get_learned_sql_store` is not importable.

- [ ] **Step 14: Add golden capture to the approve path of `confidence_gate_node`.**
  In `openchatbi/text2sql/generate_sql.py`, add the S3 store accessor import near the top (after L26):
  ```python
  from openchatbi.text2sql.data import get_learned_sql_store
  ```
  (S3 exposes `get_learned_sql_store()` returning the singleton `LearnedSQLStore` or `None`.) Add a helper inside `create_sql_nodes` (just before `confidence_gate_node`):
  ```python
      def _capture_golden_sql(state: SQLGraphState) -> None:
          """Dual-write an approved SQL: S3 vector store + durable YAML (mandatory)."""
          try:
              cfg = config.get()
          except ValueError:
              return
          if not bool(getattr(cfg, "enable_golden_sql", False)):
              return
          question = state.get("rewrite_question", "")
          sql_query = state.get("sql", "").strip()
          tables = [d["table"] for d in state.get("tables", []) if isinstance(d, dict) and d.get("table")]
          if not question or not sql_query:
              return
          namespace = getattr(cfg, "golden_sql_namespace", "global")
          # 1) runtime vector store (S3) — under the store's own lock.
          try:
              store = get_learned_sql_store()
              if store is not None:
                  store.add_golden_sql(question, sql_query, tables)
          except Exception as e:
              log(f"Golden SQL vector write failed: {str(e)}")
          # 2) durable YAML append (de-dup, not overwrite) — both writes are mandatory.
          try:
              cfg.catalog_store.append_sql_example(question, sql_query, tables, source="golden")
          except Exception as e:
              log(f"Golden SQL durable write failed: {str(e)}")
  ```
  Then in `confidence_gate_node`, call it on every approve (both the auto-approve and the human-approve paths). Replace the final returns of `confidence_gate_node` so approval triggers capture:
  ```python
          score = state.get("sql_confidence", 1.0)
          if not enabled or score is None or score >= threshold:
              _capture_golden_sql(state)
              return {"human_sql_decision": "approve"}
          reasons = state.get("confidence_reasons", [])
          feedback = interrupt(
              {
                  "text": f"Low-confidence SQL ({score:.2f}). Reasons: {'; '.join(reasons) or 'n/a'}. Approve?",
                  "buttons": ["approve", "reject", "edit"],
                  "sql": state.get("sql", ""),
              }
          )
          decision = feedback if isinstance(feedback, str) else (feedback or {}).get("decision", "approve")
          if decision == "edit":
              edited = feedback.get("sql") if isinstance(feedback, dict) else None
              if edited:
                  return {"human_sql_decision": "edit", "sql": edited}
          if decision == "approve":
              _capture_golden_sql(state)
          return {"human_sql_decision": decision}
  ```
- [ ] **Step 15: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestGoldenCaptureOnApprove -v`
  Expected: PASS — golden capture dual-writes only when `enable_golden_sql=True`; the S3 `add_golden_sql` and the durable `append_sql_example` both fire on approve.

- [ ] **Step 16: Commit.**
  Run: `git add openchatbi/text2sql/generate_sql.py tests/test_golden_sql_capture.py && git commit -m "feat(hitl): capture golden SQL on approve (S3 add_golden_sql + durable append, dual-write)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

- [ ] **Step 17: Write failing test for the `sql_examples` KB branch and the `business` branch fix in `search_knowledge`.**
  Append to `tests/test_golden_sql_capture.py`:
  ```python
  class TestSearchKnowledgeSqlExamples:
      def test_sql_examples_branch_returns_retrieved_examples(self):
          from openchatbi.tool import search_knowledge as sk

          fake_store = Mock()
          fake_store.retrieve.return_value = [
              ("how many users", "SELECT COUNT(*) FROM users", ["users"]),
          ]
          with patch.object(sk, "get_learned_sql_store", return_value=fake_store):
              result = sk.search_knowledge.invoke(
                  {
                      "reasoning": "need examples",
                      "query_list": ["user count"],
                      "knowledge_bases": ["sql_examples"],
                      "with_table_list": False,
                  }
              )
          assert "sql_examples" in result
          assert "SELECT COUNT(*) FROM users" in result["sql_examples"]
          fake_store.retrieve.assert_called()

      def test_business_branch_now_implemented(self):
          # Previously 'business' was documented but never branched; it must now
          # at least return a (possibly empty) keyed entry, not silently drop.
          from openchatbi.tool import search_knowledge as sk

          with patch.object(sk, "get_learned_sql_store", return_value=None):
              result = sk.search_knowledge.invoke(
                  {
                      "reasoning": "biz",
                      "query_list": ["revenue"],
                      "knowledge_bases": ["business"],
                      "with_table_list": False,
                  }
              )
          assert "business" in result
  ```
- [ ] **Step 18: Run the test — expect FAIL.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestSearchKnowledgeSqlExamples -v`
  Expected: FAIL — `search_knowledge` only implements the `columns` branch; `sql_examples` and `business` are absent from the result dict.

- [ ] **Step 19: Add the `sql_examples` and `business` branches to `search_knowledge`.**
  In `openchatbi/tool/search_knowledge.py`, add the import (after L7):
  ```python
  from openchatbi.text2sql.data import get_learned_sql_store
  ```
  Extend the `SearchInput.knowledge_bases` docstring (L16-20) to list the new option:
  ```python
      knowledge_bases: list[str] = Field(
          description="""Knowledge bases to search, options are:
              - `"columns"`: The description, alias of columns, including dimensions and metrics.
              - `"business"`: The business knowledge.
              - `"sql_examples"`: Validated example Question->SQL pairs (golden / learned)."""
      )
  ```
  Replace the body of `search_knowledge` (L34-39) with all three branches:
  ```python
      log(f"Search knowledge, query_list={query_list}, knowledge_bases={knowledge_bases}, reasoning={reasoning}")
      final_results = {}
      if "columns" in knowledge_bases:
          column_results = _search_column_from_catalog(query_list, with_table_list)
          final_results["columns"] = f"# Relevant Columns and Description:\n{column_results}"
      if "business" in knowledge_bases:
          # Business knowledge has no dedicated index yet; surface columns context
          # so the branch is no longer silently dropped (was docstring-only before).
          business_results = _search_column_from_catalog(query_list, with_table_list)
          final_results["business"] = f"# Relevant Business Knowledge:\n{business_results}"
      if "sql_examples" in knowledge_bases:
          final_results["sql_examples"] = _search_sql_examples(query_list)
      return final_results
  ```
  Add the helper at the end of the module:
  ```python
  def _search_sql_examples(query_list: list[str]) -> str:
      """Retrieve top-k validated Question->SQL examples from the learned SQL store (S3)."""
      store = get_learned_sql_store()
      if store is None:
          return "# Relevant SQL Examples:\n(no learned SQL store available)"
      seen: set[str] = set()
      blocks: list[str] = []
      for query in query_list:
          for question, sql, _tables in store.retrieve(query, k=5):
              if question in seen:
                  continue
              seen.add(question)
              blocks.append(f"<example>\nQ: {question}\nA: {sql}\n</example>")
      body = "\n".join(blocks) if blocks else "(no matching examples)"
      return f"# Relevant SQL Examples:\n{body}"
  ```
- [ ] **Step 20: Run the test — expect PASS.**
  Run: `uv run pytest tests/test_golden_sql_capture.py::TestSearchKnowledgeSqlExamples -v`
  Expected: PASS.

- [ ] **Step 21: Run the existing search_knowledge suite to confirm the `columns` branch is unchanged.**
  Run: `uv run pytest tests/test_tools_search_knowledge.py tests/test_golden_sql_capture.py -v`
  Expected: PASS — the `columns` branch behavior and result key are preserved; the new branches are additive.

- [ ] **Step 22: Commit.**
  Run: `git add openchatbi/tool/search_knowledge.py tests/test_golden_sql_capture.py && git commit -m "feat(hitl): add sql_examples KB branch + implement business branch in search_knowledge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

## Memory(Tasks 13-15)

### Task 13: Memory scoring + config foundation (`memory_scoring.py`, `memory_config.py`, `Config.memory_config`)

**Files:**
- Create: `openchatbi/memory_scoring.py`
- Create: `openchatbi/memory_config.py`
- Create: `tests/test_memory_scoring.py`
- Create: `tests/test_memory_config.py`
- Modify: `openchatbi/config_loader.py` (declare `memory_config` field on `Config`, after `context_config` at L83)

- [ ] **Step 1: Write failing test for `decay_factor` and `composite_score`.**
  Create `tests/test_memory_scoring.py`:
  ```python
  """Tests for memory scoring (decay + composite ranking)."""

  from datetime import datetime, timedelta, timezone

  from openchatbi.memory_config import MemoryConfig
  from openchatbi.memory_scoring import (
      bump_on_access,
      composite_score,
      decay_factor,
  )


  def _iso(days_ago: float) -> str:
      return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


  def test_decay_factor_at_zero_age_is_one():
      now = datetime.now(timezone.utc)
      assert decay_factor(now.isoformat(), half_life_days=30.0, now=now) == 1.0


  def test_decay_factor_at_half_life_is_half():
      now = datetime.now(timezone.utc)
      last_used = (now - timedelta(days=30)).isoformat()
      assert abs(decay_factor(last_used, half_life_days=30.0, now=now) - 0.5) < 1e-6


  def test_decay_factor_bad_timestamp_falls_back_to_one():
      assert decay_factor("not-a-date", half_life_days=30.0) == 1.0


  def test_composite_score_blends_similarity_importance_decay_usecount():
      cfg = MemoryConfig(importance_decay_half_life_days=30.0)
      now = datetime.now(timezone.utc)
      fresh = composite_score(0.8, 1.0, now.isoformat(), 5, cfg)
      stale = composite_score(0.8, 1.0, (now - timedelta(days=90)).isoformat(), 5, cfg)
      # Fresher memory must outrank a stale one with identical similarity/importance.
      assert fresh > stale


  def test_composite_score_higher_importance_wins_at_equal_similarity():
      cfg = MemoryConfig()
      iso = datetime.now(timezone.utc).isoformat()
      assert composite_score(0.6, 2.0, iso, 1, cfg) > composite_score(0.6, 1.0, iso, 1, cfg)


  def test_bump_on_access_increments_use_count_and_stamps_last_used():
      meta = {"use_count": 2}
      out = bump_on_access(meta)
      assert out["use_count"] == 3
      assert "last_used" in out
      # original dict not mutated in place
      assert meta["use_count"] == 2
  ```

- [ ] **Step 2: Run the test — expect failure (no module yet).**
  Run: `uv run pytest tests/test_memory_scoring.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'openchatbi.memory_config'` / `openchatbi.memory_scoring`.

- [ ] **Step 3: Create `openchatbi/memory_config.py`.**
  ```python
  """Configuration for memory & pattern-learning settings (mirrors context_config.py)."""

  from dataclasses import dataclass

  from openchatbi import config


  @dataclass
  class MemoryConfig:
      """Configuration class for memory pattern learning and decay reranking.

      All behavior-changing flags default OFF to guarantee zero regression.
      """

      # Auto-capture of successful SQL into the LearnedSQLStore (source='auto').
      enable_pattern_memory: bool = False
      # Decay/importance reranking of langmem long-term user memory.
      enable_memory_decay_rerank: bool = False

      # Namespace for captured SQL patterns (schema-level only; never PII).
      pattern_scope: str = "global"
      # Half-life (days) controlling exponential recency decay.
      importance_decay_half_life_days: float = 30.0
      # Drop retrieved items whose composite score is below this floor.
      min_retrieval_score: float = 0.2
      # Cap on blended few-shot examples injected per query.
      max_patterns_per_query: int = 5
      # How often prune_stale may run (hours).
      prune_interval_hours: int = 24


  def get_memory_config() -> MemoryConfig:
      """Get the current memory configuration.

      Loads `memory_config` from the main config system, falling back to defaults
      when the config system is unavailable or the field is unset.

      Returns:
          MemoryConfig: The current memory configuration.
      """
      try:
          main_config = config.get()

          if hasattr(main_config, "memory_config") and main_config.memory_config:
              memory_config_dict = main_config.memory_config
              memory_config = MemoryConfig()
              for key, value in memory_config_dict.items():
                  if hasattr(memory_config, key):
                      setattr(memory_config, key, value)
              return memory_config
      except (ImportError, ValueError, AttributeError):
          pass

      return MemoryConfig()
  ```

- [ ] **Step 4: Create `openchatbi/memory_scoring.py`.**
  ```python
  """Scoring helpers shared by S3 SQL-pattern retrieval and langmem long-term rerank."""

  import math
  from datetime import datetime, timezone
  from typing import Any


  def _parse_iso(ts: str) -> datetime | None:
      """Parse an ISO-8601 timestamp, returning None on any failure."""
      try:
          dt = datetime.fromisoformat(ts)
      except (ValueError, TypeError):
          return None
      if dt.tzinfo is None:
          dt = dt.replace(tzinfo=timezone.utc)
      return dt


  def decay_factor(last_used_iso: str, half_life_days: float, now: datetime | None = None) -> float:
      """Exponential recency decay: exp(-ln2 * age_days / half_life_days).

      Returns 1.0 for unparseable timestamps or non-positive half_life (no decay).
      """
      if half_life_days <= 0:
          return 1.0
      last_used = _parse_iso(last_used_iso)
      if last_used is None:
          return 1.0
      now = now or datetime.now(timezone.utc)
      if now.tzinfo is None:
          now = now.replace(tzinfo=timezone.utc)
      age_days = max(0.0, (now - last_used).total_seconds() / 86400.0)
      return math.exp(-math.log(2) * age_days / half_life_days)


  def composite_score(
      similarity: float,
      importance: float,
      last_used_iso: str,
      use_count: int,
      cfg: Any,
  ) -> float:
      """Blend similarity x importance x recency-decay, lightly boosted by use_count.

      Args:
          similarity: Retrieval similarity in [0, 1].
          importance: Provenance weight (golden > auto).
          last_used_iso: ISO timestamp of last access.
          use_count: Number of times this memory has been used.
          cfg: A MemoryConfig (reads importance_decay_half_life_days).

      Returns:
          float: A non-negative composite ranking score.
      """
      half_life = getattr(cfg, "importance_decay_half_life_days", 30.0)
      decay = decay_factor(last_used_iso, half_life)
      usage_boost = 1.0 + math.log1p(max(0, int(use_count or 0))) * 0.1
      return float(similarity) * float(importance) * decay * usage_boost


  def bump_on_access(meta: dict) -> dict:
      """Return a copy of `meta` with use_count+=1 and last_used=now (UTC ISO)."""
      out = dict(meta)
      out["use_count"] = int(out.get("use_count", 0) or 0) + 1
      out["last_used"] = datetime.now(timezone.utc).isoformat()
      return out


  def prune_stale(store: Any, namespace: str, cfg: Any) -> int:
      """Remove items whose composite recency-decay drops below cfg.min_retrieval_score.

      Iterates `store.search((namespace,))` items, computing a recency-only decay
      score (similarity treated as 1.0) and deleting those below the floor. Returns
      the number of pruned items. Best-effort: store errors are swallowed.

      Args:
          store: A langgraph BaseStore-like object with search()/delete().
          namespace: The top-level namespace segment to prune.
          cfg: A MemoryConfig (reads min_retrieval_score, importance_decay_half_life_days).

      Returns:
          int: Count of pruned items.
      """
      pruned = 0
      floor = getattr(cfg, "min_retrieval_score", 0.2)
      half_life = getattr(cfg, "importance_decay_half_life_days", 30.0)
      try:
          items = store.search((namespace,))
      except Exception:
          return 0
      for item in items:
          value = getattr(item, "value", None) or {}
          last_used = value.get("last_used", "")
          importance = float(value.get("importance", 1.0) or 1.0)
          score = importance * decay_factor(last_used, half_life)
          if score < floor:
              try:
                  store.delete(getattr(item, "namespace", (namespace,)), getattr(item, "key", ""))
                  pruned += 1
              except Exception:
                  continue
      return pruned
  ```

- [ ] **Step 5: Run the scoring test — expect PASS.**
  Run: `uv run pytest tests/test_memory_scoring.py -v`
  Expected: PASS — all 6 tests green.

- [ ] **Step 6: Write failing test for `Config.memory_config` declaration.**
  Create `tests/test_memory_config.py`:
  ```python
  """Tests for MemoryConfig loading via the main Config (pydantic field declaration)."""

  from unittest.mock import MagicMock

  from openchatbi.config_loader import Config, ConfigLoader
  from openchatbi.memory_config import MemoryConfig, get_memory_config


  def test_config_declares_memory_config_field():
      # pydantic BaseModel silently drops undeclared fields; this proves it is declared.
      assert "memory_config" in Config.model_fields


  def test_memory_config_defaults_off():
      cfg = MemoryConfig()
      assert cfg.enable_pattern_memory is False
      assert cfg.enable_memory_decay_rerank is False


  def test_get_memory_config_reads_from_main_config():
      config_dict = {
          "organization": "Test Company",
          "dialect": "presto",
          "default_llm": MagicMock(),
          "embedding_model": MagicMock(),
          "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
          "memory_config": {"enable_pattern_memory": True, "max_patterns_per_query": 3},
      }
      loader = ConfigLoader()
      loader.set(config_dict)

      mc = get_memory_config()
      assert mc.enable_pattern_memory is True
      assert mc.max_patterns_per_query == 3
      # unspecified keys keep their defaults
      assert mc.enable_memory_decay_rerank is False


  def test_get_memory_config_defaults_when_unset():
      config_dict = {
          "organization": "Test Company",
          "dialect": "presto",
          "default_llm": MagicMock(),
          "embedding_model": MagicMock(),
          "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
      }
      loader = ConfigLoader()
      loader.set(config_dict)

      mc = get_memory_config()
      assert mc.enable_pattern_memory is False
  ```

- [ ] **Step 7: Run the config test — expect failure.**
  Run: `uv run pytest tests/test_memory_config.py -v`
  Expected: FAIL — `test_config_declares_memory_config_field` fails (`memory_config` not in `Config.model_fields`); `test_get_memory_config_reads_from_main_config` fails because the undeclared field is silently dropped (`extra='ignore'`), so `mc.enable_pattern_memory` is False.

- [ ] **Step 8: Declare `memory_config` on `Config` (after `context_config` at L83).**
  In `openchatbi/config_loader.py`, edit:
  ```python
      # Context Management Configuration
      context_config: dict[str, Any] = {}

      # Memory & Pattern Learning Configuration (mirrors context_config; see memory_config.py)
      memory_config: dict[str, Any] = {}
  ```

- [ ] **Step 9: Run the config test — expect PASS.**
  Run: `uv run pytest tests/test_memory_config.py -v`
  Expected: PASS — all 4 tests green.

- [ ] **Step 10: Run the full new suite + config regression to confirm zero breakage.**
  Run: `uv run pytest tests/test_memory_scoring.py tests/test_memory_config.py tests/test_config_loader.py -v`
  Expected: PASS — new tests green, existing config_loader tests unchanged.

- [ ] **Step 11: Commit.**
  Run:
  ```
  git add openchatbi/memory_scoring.py openchatbi/memory_config.py openchatbi/config_loader.py tests/test_memory_scoring.py tests/test_memory_config.py && git commit -m "feat(memory): add memory_scoring + memory_config foundation (defaults OFF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

### Task 14: Auto-capture + blended retrieval in `generate_sql.py` (gated by S2 confidence)

**Files:**
- Modify: `openchatbi/text2sql/generate_sql.py` (`create_sql_nodes` signature L136-138; `_get_relevant_sql_examples_prompt` L187-208 incl. `used_tables⊆selected` filter @L205; `execute_sql_node` SQL_SUCCESS path L344-356)
- Modify: `openchatbi/text2sql/sql_graph.py` (pass `memory_store` handle into `create_sql_nodes`, call site L82-87)
- Create: `tests/test_pattern_memory_capture.py`

> Depends on Task 13 (`memory_scoring`/`memory_config`), S2 (`SimpleSQLEvaluator` in `openchatbi/text2sql/confidence.py`) and S3 (`LearnedSQLStore` exposing `add(...)`/`retrieve(...)`). Both default OFF — auto-capture only fires when `enable_pattern_memory` is True AND the S2 gate passes.

- [ ] **Step 1: Write failing test for blended retrieval + gated auto-capture.**
  Create `tests/test_pattern_memory_capture.py`:
  ```python
  """Tests for auto-capture into LearnedSQLStore and blended SQL-example retrieval."""

  from unittest.mock import MagicMock, patch

  from langchain_core.language_models import FakeListChatModel

  from openchatbi.constants import SQL_SUCCESS
  from openchatbi.text2sql.generate_sql import create_sql_nodes


  def _make_nodes(catalog, store, mock_llm):
      # create_sql_nodes now accepts an optional learned_sql_store handle (last positional/kw arg).
      return create_sql_nodes(
          mock_llm,
          catalog,
          dialect="presto",
          visualization_mode=None,
          learned_sql_store=store,
      )


  def test_create_sql_nodes_accepts_store_handle(mock_catalog_store, mock_llm):
      nodes = _make_nodes(mock_catalog_store, MagicMock(), mock_llm)
      assert len(nodes) == 4  # signature unchanged in arity of returned nodes


  @patch("openchatbi.text2sql.generate_sql.get_memory_config")
  def test_auto_capture_disabled_by_default(mock_get_cfg, mock_catalog_store, mock_llm):
      cfg = MagicMock()
      cfg.enable_pattern_memory = False
      mock_get_cfg.return_value = cfg
      store = MagicMock()

      _gen, execute_sql_node, _regen, _viz = _make_nodes(mock_catalog_store, store, mock_llm)

      with patch("openchatbi.text2sql.generate_sql._execute_sql_for_node") as mock_exec:
          mock_exec.return_value = ({"row_count": 1, "columns": ["c"]}, "c\n1\n")
          state = {"sql": "SELECT 1", "rewrite_question": "q", "tables": [{"table": "t", "columns": []}],
                   "sql_retry_count": 0}
          out = execute_sql_node(state)

      assert out["sql_execution_result"] == SQL_SUCCESS
      store.add.assert_not_called()  # flag OFF -> never captured


  @patch("openchatbi.text2sql.generate_sql.SimpleSQLEvaluator")
  @patch("openchatbi.text2sql.generate_sql.get_memory_config")
  def test_auto_capture_fires_when_enabled_and_gate_passes(
      mock_get_cfg, mock_evaluator_cls, mock_catalog_store, mock_llm
  ):
      cfg = MagicMock()
      cfg.enable_pattern_memory = True
      cfg.pattern_scope = "global"
      cfg.sql_confidence_threshold = 0.7
      mock_get_cfg.return_value = cfg

      verdict = MagicMock()
      verdict.score = 0.9
      mock_evaluator_cls.return_value.evaluate.return_value = verdict

      store = MagicMock()
      _gen, execute_sql_node, _regen, _viz = _make_nodes(mock_catalog_store, store, mock_llm)

      with patch("openchatbi.text2sql.generate_sql._execute_sql_for_node") as mock_exec:
          mock_exec.return_value = ({"row_count": 1, "columns": ["c"]}, "c\n1\n")
          state = {"sql": "SELECT 1", "rewrite_question": "how many", "tables": [{"table": "t", "columns": []}],
                   "sql_retry_count": 0}
          out = execute_sql_node(state)

      assert out["sql_execution_result"] == SQL_SUCCESS
      store.add.assert_called_once()
      _, kwargs = store.add.call_args
      assert kwargs["source"] == "auto"
      assert kwargs["namespace"] == "global"


  @patch("openchatbi.text2sql.generate_sql.SimpleSQLEvaluator")
  @patch("openchatbi.text2sql.generate_sql.get_memory_config")
  def test_auto_capture_skipped_when_gate_fails(
      mock_get_cfg, mock_evaluator_cls, mock_catalog_store, mock_llm
  ):
      cfg = MagicMock()
      cfg.enable_pattern_memory = True
      cfg.pattern_scope = "global"
      cfg.sql_confidence_threshold = 0.7
      mock_get_cfg.return_value = cfg

      verdict = MagicMock()
      verdict.score = 0.3  # below threshold -> "success != correct", do not poison pool
      mock_evaluator_cls.return_value.evaluate.return_value = verdict

      store = MagicMock()
      _gen, execute_sql_node, _regen, _viz = _make_nodes(mock_catalog_store, store, mock_llm)

      with patch("openchatbi.text2sql.generate_sql._execute_sql_for_node") as mock_exec:
          mock_exec.return_value = ({"row_count": 1, "columns": ["c"]}, "c\n1\n")
          state = {"sql": "SELECT 1", "rewrite_question": "q", "tables": [{"table": "t", "columns": []}],
                   "sql_retry_count": 0}
          out = execute_sql_node(state)

      assert out["sql_execution_result"] == SQL_SUCCESS
      store.add.assert_not_called()


  @patch("openchatbi.text2sql.generate_sql.get_memory_config")
  def test_blended_retrieval_uses_store_when_present(mock_get_cfg, mock_catalog_store, mock_llm):
      cfg = MagicMock()
      cfg.enable_pattern_memory = True
      cfg.max_patterns_per_query = 5
      mock_get_cfg.return_value = cfg

      store = MagicMock()
      store.retrieve.return_value = [("how many users", "SELECT COUNT(*) FROM test_table", ["test_table"])]
      gen, _exec, _regen, _viz = _make_nodes(mock_catalog_store, store, mock_llm)

      # generate_sql_node builds the prompt via _get_relevant_sql_examples_prompt internally.
      state = {"rewrite_question": "how many users", "tables": [{"table": "test_table", "columns": []}],
               "messages": []}
      gen(state)
      store.retrieve.assert_called_once()
      _, kwargs = store.retrieve.call_args
      assert kwargs.get("score_fn") is not None  # composite_score injected
  ```

- [ ] **Step 2: Run the test — expect failure.**
  Run: `uv run pytest tests/test_pattern_memory_capture.py -v`
  Expected: FAIL — `create_sql_nodes() got an unexpected keyword argument 'learned_sql_store'`; `_execute_sql_for_node` does not exist.

- [ ] **Step 3: Extend `create_sql_nodes` signature and refactor `_execute_sql` to a module-level patch point.**
  In `openchatbi/text2sql/generate_sql.py`, add imports near the existing `from openchatbi.text2sql.data import ...` line:
  ```python
  from openchatbi.memory_config import get_memory_config
  from openchatbi.memory_scoring import composite_score
  from openchatbi.text2sql.confidence import SimpleSQLEvaluator
  ```
  Change the `create_sql_nodes` signature (L136-138):
  ```python
  def create_sql_nodes(
      llm: BaseChatModel,
      catalog: CatalogStore,
      dialect: str,
      visualization_mode: str | None = "rule",
      learned_sql_store: Any | None = None,
  ) -> tuple[Callable, Callable, Callable, Callable]:
  ```
  Add a thin module-level indirection so tests can patch the executor independently of the inner closure. Just below `create_sql_nodes`'s docstring (before `visualization_service = ...`), bind the inner `_execute_sql` through a patchable seam by renaming the inner call site. Add this module-level function above `create_sql_nodes` (after `_classify_operational_error`):
  ```python
  def _execute_sql_for_node(catalog: CatalogStore, sql: str) -> tuple[dict, str]:
      """Module-level seam wrapping the catalog-bound SQL execution (patchable in tests)."""
      return _execute_sql_impl(catalog, sql)
  ```
  And extract the body of the existing inner `_execute_sql` into a module-level `_execute_sql_impl(catalog, sql)` (move it out of the closure verbatim, replacing its use of the closed-over `catalog` with the parameter):
  ```python
  def _execute_sql_impl(catalog: CatalogStore, sql: str) -> tuple[dict, str]:
      """Execute the validated SQL and return (schema_info, csv_string)."""
      limit_enabled, result_limit = _get_sql_result_limit_config()
      is_safe, reason = _validate_sql_safety(sql)
      if not is_safe:
          raise SQLSecurityError(reason)

      execute_sql = _limit_sql_query(sql, result_limit) if limit_enabled else sql
      with catalog.get_sql_engine().connect() as connection:
          result = connection.execute(text(execute_sql))
          rows = result.fetchmany(result_limit) if limit_enabled else result.fetchall()
          columns = list(result.keys())
          df = pd.DataFrame(rows, columns=columns)
          schema_info = _analyze_dataframe_schema_impl(df)
          if limit_enabled:
              schema_info["result_limit"] = result_limit
          csv_data = df.to_csv(index=False)
          connection.commit()
          return schema_info, csv_data
  ```
  Move `_analyze_dataframe_schema` out to module level as `_analyze_dataframe_schema_impl(df)` (verbatim body) and delete the now-duplicated inner `_execute_sql`/`_analyze_dataframe_schema` closures. Inside the closure, replace the old `schema_info, csv_result = _execute_sql(sql_query)` with `schema_info, csv_result = _execute_sql_for_node(catalog, sql_query)`.

- [ ] **Step 4: Add the gated auto-capture helper inside `create_sql_nodes`.**
  Inside `create_sql_nodes`, after `visualization_service = ...`, add the capture closure:
  ```python
  def _maybe_capture_pattern(state: SQLGraphState) -> None:
      """Fire-and-forget: capture (question -> SQL -> tables) into LearnedSQLStore.

      Gated by enable_pattern_memory AND the S2 confidence gate (success != correct).
      Never raises; never blocks the response.
      """
      if learned_sql_store is None:
          return
      try:
          mem_cfg = get_memory_config()
          if not getattr(mem_cfg, "enable_pattern_memory", False):
              return
          question = state.get("rewrite_question", "")
          sql_query = state.get("sql", "").strip()
          tables = [d["table"] for d in state.get("tables", [])]
          if not question or not sql_query:
              return
          # S2 gate: only promote SQL the evaluator considers correct.
          threshold = float(getattr(config.get(), "sql_confidence_threshold", 0.7))
          schema_info = state.get("schema_info", {})
          data_sample = state.get("data", "")
          verdict = SimpleSQLEvaluator().evaluate(question, sql_query, schema_info, data_sample)
          if verdict.score < threshold:
              log(f"Pattern capture skipped: confidence {verdict.score:.2f} < {threshold:.2f}")
              return
          retry_count = int(state.get("sql_retry_count", 0) or 0)
          importance = 1.0 / (1.0 + retry_count)  # first-try success weighted highest
          learned_sql_store.add(
              question,
              sql_query,
              tables,
              source="auto",
              importance=importance,
              namespace=getattr(mem_cfg, "pattern_scope", "global"),
          )
      except Exception as e:  # never let capture break the response
          log(f"Pattern capture error (ignored): {e}")
  ```

- [ ] **Step 5: Call the capture helper on the SQL_SUCCESS path.**
  In `execute_sql_node`, in the `try` block right after building `result` and before `return {...}` (L350-356), invoke capture so it runs only on success:
  ```python
          result = f"```sql\n{sql_query}\n```\n{result_label}:\n```csv\n{csv_result}\n```"
          _maybe_capture_pattern({**state, "schema_info": schema_info, "data": csv_result})
          return {
              "sql_execution_result": SQL_SUCCESS,
              "schema_info": schema_info,
              "data": csv_result,
              "messages": [AIMessage(result)],
          }
  ```

- [ ] **Step 6: Make `_get_relevant_sql_examples_prompt` blend via the store, relaxing the `used_tables⊆selected` filter.**
  Replace the body of `_get_relevant_sql_examples_prompt` (L187-208). When a store is wired and pattern memory is enabled, use `LearnedSQLStore.retrieve(score_fn=composite_score)` and relax the strict subset filter to a *soft* one (keep examples whose used_tables intersect the selected tables, capped by `max_patterns_per_query`); otherwise fall back to the legacy static retriever with the original strict filter (zero regression):
  ```python
  def _get_relevant_sql_examples_prompt(question, tables_columns: list[dict[str, Any]]) -> str:
      """Retrieve relevant SQL examples for the question and selected tables.

      Blends static + golden + auto patterns via LearnedSQLStore when enabled;
      otherwise preserves the legacy static-retriever behavior (strict subset filter).
      """
      tables = [d["table"] for d in tables_columns]
      mem_cfg = get_memory_config()

      if learned_sql_store is not None and getattr(mem_cfg, "enable_pattern_memory", False):
          cap = int(getattr(mem_cfg, "max_patterns_per_query", 5))
          retrieved = learned_sql_store.retrieve(
              question, k=max(cap * 2, 10), score_fn=composite_score
          )
          examples = []
          for ex_question, example_sql, used_tables in retrieved:
              # Soft filter: keep when patterns touch any selected table (relaxed from strict subset).
              if not used_tables or any(t in tables for t in used_tables):
                  examples.append(f"<example>\nQ: {ex_question}\nA: {example_sql}\n</example>\n")
              if len(examples) >= cap:
                  break
          log(f"Blended examples (store): {examples}")
          return "\n".join(examples)

      # Legacy path: static retriever + strict subset filter (unchanged behavior).
      relevant_questions = sql_example_retriever.invoke(question)
      examples = []
      for relevant_document in relevant_questions:
          q = relevant_document.page_content
          example_sql, used_tables = sql_example_dicts[q]
          if all(table in tables for table in used_tables):
              examples.append(f"<example>\nQ: {q}\nA: {example_sql}\n</example>\n")
      log(f"Examples using selected tables: {examples}")
      return "\n".join(examples)
  ```

- [ ] **Step 7: Wire the store handle through `build_sql_graph` into `create_sql_nodes`.**
  In `openchatbi/text2sql/sql_graph.py`, import the store accessor near the top imports:
  ```python
  from openchatbi.text2sql.data import learned_sql_store
  ```
  Update the `create_sql_nodes` call (L82-87) to forward the handle:
  ```python
      generate_sql_node, execute_sql_node, regenerate_sql_node, generate_visualization_node = create_sql_nodes(
          get_text2sql_llm(llm_provider),
          catalog,
          dialect=config.get().dialect,
          visualization_mode=config.get().visualization_mode,
          learned_sql_store=learned_sql_store,
      )
  ```
  > Note: `learned_sql_store` is the module-level `LearnedSQLStore` instance created by S3 (Task in cluster S3 adds it to `data.py`). It is `None` when no catalog/embedding is configured, so the legacy retrieval path stays active and capture is a no-op.

- [ ] **Step 8: Run the new test — expect PASS.**
  Run: `uv run pytest tests/test_pattern_memory_capture.py -v`
  Expected: PASS — all 6 tests green.

- [ ] **Step 9: Run the existing generate_sql + sql_graph regression suites.**
  Run: `uv run pytest tests/test_generate_sql.py tests/test_sql_graph.py -v`
  Expected: PASS — the refactor of `_execute_sql`/`_analyze_dataframe_schema` to module-level seams and the default-OFF gating preserve all existing assertions (error_type strings and SQL_* codes unchanged).

- [ ] **Step 10: Commit.**
  Run:
  ```
  git add openchatbi/text2sql/generate_sql.py openchatbi/text2sql/sql_graph.py tests/test_pattern_memory_capture.py && git commit -m "feat(memory): gated SQL pattern auto-capture + blended retrieval via LearnedSQLStore

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

### Task 15: langmem decay rerank in `tool/memory.py` (`enable_memory_decay_rerank`, default OFF)

**Files:**
- Modify: `openchatbi/tool/memory.py` (`get_memory_tools` L168-187 — wrap `search_memory_tool` re-rank; stamp `importance/last_used/use_count` on `manage_memory` writes)
- Create: `tests/test_memory_decay_rerank.py`

> Depends on Task 13 (`memory_scoring.composite_score`/`bump_on_access`, `memory_config.get_memory_config`). Default OFF: when `enable_memory_decay_rerank` is False the original langmem tools are returned untouched.

- [ ] **Step 1: Write failing test for the decay-rerank wrapper and write-stamping.**
  Create `tests/test_memory_decay_rerank.py`:
  ```python
  """Tests for langmem decay reranking + importance/last_used/use_count stamping."""

  from datetime import datetime, timedelta, timezone
  from unittest.mock import Mock, patch

  import pytest
  from langchain_core.language_models import FakeListChatModel

  pytest.importorskip("pysqlite3", reason="pysqlite3 not available")

  from openchatbi.tool.memory import (  # noqa: E402
      _rerank_search_results,
      _stamp_memory_value,
      get_memory_tools,
  )


  def _item(text, days_ago, importance=1.0, use_count=0, score=0.5):
      it = Mock()
      it.value = {
          "text": text,
          "importance": importance,
          "use_count": use_count,
          "last_used": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
      }
      it.score = score
      return it


  def test_rerank_orders_by_composite_score():
      fresh = _item("fresh", days_ago=1, score=0.5)
      stale = _item("stale", days_ago=200, score=0.5)
      out = _rerank_search_results([stale, fresh])
      assert out[0].value["text"] == "fresh"  # recency-decayed composite floats fresh up


  def test_rerank_tolerates_plain_dicts():
      a = {"value": {"text": "a", "last_used": datetime.now(timezone.utc).isoformat()}, "score": 0.9}
      b = {"value": {"text": "b", "last_used": (datetime.now(timezone.utc) - timedelta(days=300)).isoformat()}, "score": 0.9}
      out = _rerank_search_results([b, a])
      assert out[0]["value"]["text"] == "a"


  def test_stamp_memory_value_adds_provenance_fields():
      stamped = _stamp_memory_value({"text": "remember X"})
      assert stamped["importance"] == 1.0
      assert stamped["use_count"] == 0
      assert "last_used" in stamped


  @patch("openchatbi.tool.memory.get_memory_config")
  @patch("openchatbi.tool.memory.create_manage_memory_tool")
  @patch("openchatbi.tool.memory.create_search_memory_tool")
  @patch("openchatbi.tool.memory.get_sync_memory_store")
  def test_rerank_disabled_by_default_returns_raw_tools(
      mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg
  ):
      cfg = Mock()
      cfg.enable_memory_decay_rerank = False
      mock_get_cfg.return_value = cfg
      mock_get_store.return_value = Mock()
      raw_search = Mock()
      raw_manage = Mock()
      mock_search_tool.return_value = raw_search
      mock_manage_tool.return_value = raw_manage

      tools = get_memory_tools(FakeListChatModel(responses=["x"]), sync_mode=True)
      # default OFF: the original langmem tool objects are returned unwrapped
      assert tools[0] is raw_manage
      assert tools[1] is raw_search


  @patch("openchatbi.tool.memory.get_memory_config")
  @patch("openchatbi.tool.memory.create_manage_memory_tool")
  @patch("openchatbi.tool.memory.create_search_memory_tool")
  @patch("openchatbi.tool.memory.get_sync_memory_store")
  def test_rerank_enabled_wraps_search_tool(
      mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg
  ):
      cfg = Mock()
      cfg.enable_memory_decay_rerank = True
      mock_get_cfg.return_value = cfg
      mock_get_store.return_value = Mock()
      mock_search_tool.return_value = Mock()
      mock_manage_tool.return_value = Mock()

      tools = get_memory_tools(FakeListChatModel(responses=["x"]), sync_mode=True)
      # wrapped tools are new StructuredTool instances, not the raw mocks
      assert tools[0] is not mock_manage_tool.return_value
      assert tools[1] is not mock_search_tool.return_value
  ```

- [ ] **Step 2: Run the test — expect failure.**
  Run: `uv run pytest tests/test_memory_decay_rerank.py -v`
  Expected: FAIL — `ImportError: cannot import name '_rerank_search_results'` / `_stamp_memory_value` from `openchatbi.tool.memory`.

- [ ] **Step 3: Add the rerank + stamping helpers and imports to `tool/memory.py`.**
  In `openchatbi/tool/memory.py`, add imports after the existing `from langchain_core.tools import StructuredTool` block:
  ```python
  from langchain_core.tools import StructuredTool, tool

  from openchatbi.memory_config import get_memory_config
  from openchatbi.memory_scoring import bump_on_access, composite_score
  ```
  Add the helpers above `get_memory_tools` (after `StructuredToolWithRequired`):
  ```python
  def _item_value(item: Any) -> dict:
      """Extract the value dict from a langgraph Item or a plain dict result."""
      if isinstance(item, dict):
          return item.get("value", {}) or {}
      return getattr(item, "value", {}) or {}


  def _item_base_score(item: Any) -> float:
      """Extract the retrieval similarity/score from an Item or dict, defaulting to 1.0."""
      if isinstance(item, dict):
          return float(item.get("score", 1.0) or 1.0)
      return float(getattr(item, "score", 1.0) or 1.0)


  def _rerank_search_results(items: list) -> list:
      """Re-rank langmem search results by composite_score(similarity, importance, decay, use_count)."""
      cfg = get_memory_config()

      def _key(item: Any) -> float:
          value = _item_value(item)
          return composite_score(
              _item_base_score(item),
              float(value.get("importance", 1.0) or 1.0),
              value.get("last_used", ""),
              int(value.get("use_count", 0) or 0),
              cfg,
          )

      return sorted(items, key=_key, reverse=True)


  def _stamp_memory_value(value: dict) -> dict:
      """Stamp importance/last_used/use_count provenance on a memory write payload."""
      from datetime import datetime, timezone

      out = dict(value)
      out.setdefault("importance", 1.0)
      out.setdefault("use_count", 0)
      out.setdefault("last_used", datetime.now(timezone.utc).isoformat())
      return out
  ```

- [ ] **Step 4: Wrap the tools in `get_memory_tools` when the flag is ON.**
  In `get_memory_tools` (L168-187), after the existing `manage_memory_tool`/`search_memory_tool` creation and the `BaseChatOpenAI` wrapping, gate the rerank/stamp wrapping behind the flag, then return:
  ```python
      if isinstance(llm, BaseChatOpenAI):
          manage_memory_tool = StructuredToolWithRequired(manage_memory_tool)
          search_memory_tool = StructuredToolWithRequired(search_memory_tool)

      mem_cfg = get_memory_config()
      if not getattr(mem_cfg, "enable_memory_decay_rerank", False):
          return [manage_memory_tool, search_memory_tool]

      _raw_search = search_memory_tool
      _raw_manage = manage_memory_tool

      @tool("search_memory", description=getattr(_raw_search, "description", "Search long-term memory."))
      def reranked_search_memory(query: str) -> Any:
          """Search long-term memory, re-ranked by importance/recency decay."""
          results = _raw_search.invoke({"query": query})
          if isinstance(results, list):
              return _rerank_search_results(results)
          return results

      @tool("manage_memory", description=getattr(_raw_manage, "description", "Create or update long-term memory."))
      def stamped_manage_memory(content: str) -> Any:
          """Create/update long-term memory, stamping importance/last_used/use_count provenance."""
          return _raw_manage.invoke({"content": _stamp_memory_value({"text": content})})

      return [stamped_manage_memory, reranked_search_memory]
  ```

- [ ] **Step 5: Run the new test — expect PASS.**
  Run: `uv run pytest tests/test_memory_decay_rerank.py -v`
  Expected: PASS — all 6 tests green.

- [ ] **Step 6: Run the existing memory suite to confirm zero regression.**
  Run: `uv run pytest tests/test_memory.py -v`
  Expected: PASS — default-OFF path returns the original langmem tools so `test_get_memory_tools_sync_mode` / `test_get_memory_tools_with_openai_llm` assertions (which compare against the raw mocks) remain green.

- [ ] **Step 7: Commit.**
  Run:
  ```
  git add openchatbi/tool/memory.py tests/test_memory_decay_rerank.py && git commit -m "feat(memory): langmem decay rerank + write-stamping (enable_memory_decay_rerank, default OFF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Eval(Tasks 16-17)

### Task 16: Expand RunLedger eval corpus 1→20+ with per-prompt scripted trajectories

**Files:**
- Create: `evals/runledger/cases/c02_aggregation.yaml` … `evals/runledger/cases/c20_report.yaml` (19 new case files, alongside existing `cases/t1.yaml`)
- Create: `evals/runledger/cassettes/c02_aggregation.jsonl` … `c20_report.jsonl` (19 new cassettes)
- Modify: `evals/runledger/agent/agent.py` (replace `_stub_llm_call` L185-200 with a per-prompt scripted trajectory driver; `_build_tool_proxies` already has all 5 proxies at L105-182 — no change there)
- Modify: `evals/runledger/suite.yaml` (expand `tool_registry` L5-6 from 1→5 tools)
- Modify: `evals/runledger/tools.py` (expand `TOOLS` L16-18 from 1→5 real callables)
- Modify: `pyproject.toml` (add `eval` optional extra after the `test` block ending L106)
- Modify: `evals/runledger/README.md` (document case/cassette/protocol format)
- Create: `tests/eval/test_runledger_agent.py` (unit test for the trajectory driver — runs under pytest, hermetic)

- [ ] **Step 1: Write a failing test for the per-prompt trajectory driver.**
  The driver must key on prompt text (case-id is NOT in the JSONL protocol). Create `tests/eval/__init__.py` (empty) and `tests/eval/test_runledger_agent.py`:
  ```python
  """Unit tests for the RunLedger scripted-trajectory agent driver."""

  import importlib

  from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


  def _load_agent():
      # agent.py mutates builtins.print on import; import once and reuse.
      return importlib.import_module("evals.runledger.agent.agent")


  def test_first_turn_keys_on_prompt_aggregation():
      agent = _load_agent()
      messages = [HumanMessage(content="How many orders were placed in 2024?")]
      out = agent._scripted_llm_call(None, messages)
      assert isinstance(out, AIMessage)
      assert len(out.tool_calls) == 1
      assert out.tool_calls[0]["name"] == "text2sql"


  def test_first_turn_keys_on_prompt_knowledge():
      agent = _load_agent()
      messages = [HumanMessage(content="OpenChatBI")]
      out = agent._scripted_llm_call(None, messages)
      assert out.tool_calls[0]["name"] == "search_knowledge"


  def test_trajectory_advances_per_tool_message():
      agent = _load_agent()
      # report case: search_knowledge -> text2sql -> save_report -> final text
      prompt = "Generate a sales report for Q1 2024"
      msgs = [HumanMessage(content=prompt)]
      first = agent._scripted_llm_call(None, msgs)
      assert first.tool_calls[0]["name"] == "search_knowledge"
      msgs += [first, ToolMessage(content="ctx", tool_call_id="c1")]
      second = agent._scripted_llm_call(None, msgs)
      assert second.tool_calls[0]["name"] == "text2sql"


  def test_unknown_prompt_falls_back_to_search_knowledge():
      agent = _load_agent()
      out = agent._scripted_llm_call(None, [HumanMessage(content="totally novel question")])
      assert out.tool_calls[0]["name"] == "search_knowledge"


  def test_final_turn_emits_no_tool_calls():
      agent = _load_agent()
      prompt = "How many orders were placed in 2024?"
      # aggregation trajectory is single tool then summary
      msgs = [
          HumanMessage(content=prompt),
          AIMessage(content="", tool_calls=[{"name": "text2sql", "args": {}, "id": "c1"}]),
          ToolMessage(content="result", tool_call_id="c1"),
      ]
      out = agent._scripted_llm_call(None, msgs)
      assert out.tool_calls == []
  ```

- [ ] **Step 2: Run the test — expect failure (driver not implemented).**
  Run: `uv run pytest tests/eval/test_runledger_agent.py -v`
  Expected: FAIL — `AttributeError: module 'evals.runledger.agent.agent' has no attribute '_scripted_llm_call'`.

- [ ] **Step 3: Replace `_stub_llm_call` with a per-prompt scripted trajectory driver.**
  In `evals/runledger/agent/agent.py`, replace the `_stub_llm_call` function (L185-200) with a trajectory table keyed on prompt + a turn-counting driver. The number of `ToolMessage`s already in the message list selects which turn to emit (0 = first tool call, N = after N tool results):
  ```python
  # Each trajectory: list of turns. A turn is either a tool name (emit one tool_call)
  # or None (emit a final text answer with no tool_calls). The driver advances by the
  # count of ToolMessages already present, because the case-id is NOT in the JSONL
  # protocol — the only stable key is the user prompt text.
  _TRAJECTORIES: dict[str, list[str | None]] = {
      "OpenChatBI": ["search_knowledge", None],
      "How many orders were placed in 2024?": ["text2sql", None],
      "What is the total revenue by region?": ["text2sql", None],
      "What is the average order value per customer?": ["text2sql", None],
      "Show daily active users for the last 30 days": ["text2sql", None],
      "Join orders with customers and list top 10 spenders": ["text2sql", None],
      "Which products have orders but no shipments?": ["text2sql", None],
      "What were sales between 2024-01-01 and 2024-03-31?": ["text2sql", None],
      "Compare this month's revenue to last month": ["text2sql", None],
      "Detect anomalies in daily signup counts": ["text2sql", "run_python_code", None],
      "Plot the revenue trend for 2024": ["text2sql", "run_python_code", None],
      "What columns describe customer churn?": ["search_knowledge", None],
      "Explain the orders fact table": ["show_schema", None],
      "What does the metric DAU mean?": ["search_knowledge", None],
      "List the schema of the customers table": ["show_schema", None],
      "How many active users signed up last week?": ["text2sql", None],
      "Forecast next quarter revenue from history": ["text2sql", "run_python_code", None],
      "Break down conversion rate by channel": ["text2sql", None],
      "Generate a sales report for Q1 2024": ["search_knowledge", "text2sql", "save_report", None],
      "Summarize order volume and save it as a report": ["text2sql", "save_report", None],
  }

  # Default trajectory for any prompt not in the table (e.g. novel record-mode runs).
  _DEFAULT_TRAJECTORY: list[str | None] = ["search_knowledge", None]

  _TOOL_ARGS_BUILDERS = {
      "search_knowledge": lambda q: {
          "reasoning": "Look up relevant knowledge",
          "query_list": [q],
          "knowledge_bases": ["columns"],
          "with_table_list": False,
      },
      "show_schema": lambda q: {"reasoning": "Inspect schema", "tables": [q]},
      "text2sql": lambda q: {"reasoning": "Generate SQL", "context": q},
      "run_python_code": lambda q: {
          "reasoning": "Post-process result",
          "code": "result = df.describe()",
      },
      "save_report": lambda q: {
          "content": f"Report for: {q}",
          "title": q[:40] or "report",
          "file_format": "md",
      },
  }


  def _tool_message_count(messages: list[Any]) -> int:
      return sum(
          1 for m in messages if isinstance(m, ToolMessage) or getattr(m, "type", None) == "tool"
      )


  def _scripted_llm_call(chat_model: Any, messages: list[Any], **_kwargs: Any) -> AIMessage:
      user_text = _last_user_text(messages)
      trajectory = _TRAJECTORIES.get(user_text, _DEFAULT_TRAJECTORY)
      step = _tool_message_count(messages)
      if step >= len(trajectory) or trajectory[step] is None:
          return AIMessage(content="Here is a deterministic summary based on the tool result.", tool_calls=[])
      tool_name = trajectory[step]
      args = _TOOL_ARGS_BUILDERS[tool_name](user_text)
      call_id = f"call_{step + 1}"
      return AIMessage(
          content=f"Calling {tool_name}.",
          tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
      )
  ```

- [ ] **Step 4: Point `_configure_agent_graph` at the new driver.**
  In `evals/runledger/agent/agent.py`, change the binding in `_configure_agent_graph` (L219) from `_stub_llm_call` to `_scripted_llm_call`:
  ```python
      agent_graph.call_llm_chat_model_with_retry = _scripted_llm_call
  ```

- [ ] **Step 5: Run the test — expect pass.**
  Run: `uv run pytest tests/eval/test_runledger_agent.py -v`
  Expected: PASS — all 5 tests green; driver keys on prompt and advances by tool-message count.

- [ ] **Step 6: Expand `tool_registry` in `suite.yaml` to 5 tools.**
  In `evals/runledger/suite.yaml`, replace the single-entry `tool_registry` (L5-6):
  ```yaml
  tool_registry:
    - search_knowledge
    - show_schema
    - text2sql
    - run_python_code
    - save_report
  ```
  Also bump the budget ceiling (L13-16) so multi-step report trajectories fit:
  ```yaml
  budgets:
    max_wall_ms: 20000
    max_tool_calls: 8
    max_tool_errors: 0
  ```

- [ ] **Step 7: Register the 5 real tool callables in `tools.py`.**
  Replace the body of `evals/runledger/tools.py` (L1-18) so record-mode runs hit real OpenChatBI tools. `text2sql` has no module-level `@tool` callable (it is built per-graph via `get_sql_tools`), so register a thin record-mode proxy that returns the SQL graph context echo:
  ```python
  from __future__ import annotations

  from typing import Any

  from openchatbi.tool.run_python_code import run_python_code
  from openchatbi.tool.save_report import save_report
  from openchatbi.tool.search_knowledge import search_knowledge, show_schema


  def _invoke_tool(tool, args: dict[str, Any]) -> Any:
      return tool.invoke(args)


  def _search_knowledge(args: dict[str, Any]) -> Any:
      return _invoke_tool(search_knowledge, args)


  def _show_schema(args: dict[str, Any]) -> Any:
      return _invoke_tool(show_schema, args)


  def _run_python_code(args: dict[str, Any]) -> Any:
      return _invoke_tool(run_python_code, args)


  def _save_report(args: dict[str, Any]) -> Any:
      return _invoke_tool(save_report, args)


  def _text2sql(args: dict[str, Any]) -> Any:
      # text2sql is graph-built (get_sql_tools), not a module-level @tool callable.
      # In record mode the SQL sub-graph requires a live warehouse/LLM; the eval is
      # replay-only by default, so this proxy is a deterministic echo for recording.
      return {"sql": "", "data": "", "context": args.get("context", "")}


  TOOLS = {
      "search_knowledge": _search_knowledge,
      "show_schema": _show_schema,
      "text2sql": _text2sql,
      "run_python_code": _run_python_code,
      "save_report": _save_report,
  }
  ```

- [ ] **Step 8: Add 19 case YAMLs with `category` + `gold` (hand-written gold SQL).**
  Create one file per case in `evals/runledger/cases/`. Each carries `category`, the existing `input.prompt`/`cassette` fields, and a `gold` block (`expected_sql`, `expected_tool_trajectory`, `expected_result_contains`) consumed by the Task 17 judge — RunLedger ignores unknown top-level keys. Example `evals/runledger/cases/c02_aggregation.yaml`:
  ```yaml
  id: c02_aggregation
  category: aggregation
  description: "count aggregation over a time-filtered fact table"
  input:
    prompt: "How many orders were placed in 2024?"
  cassette: cassettes/c02_aggregation.jsonl
  gold:
    expected_sql: "SELECT COUNT(*) FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'"
    expected_tool_trajectory: ["text2sql"]
    expected_result_contains: ["count"]
  ```
  Create the remaining 18 with these (id, category, prompt, gold.expected_sql):
  - `c03_aggregation` / aggregation / "What is the total revenue by region?" / `SELECT region, SUM(revenue) AS total_revenue FROM orders GROUP BY region`
  - `c04_aggregation` / aggregation / "What is the average order value per customer?" / `SELECT customer_id, AVG(order_total) AS avg_order_value FROM orders GROUP BY customer_id`
  - `c05_aggregation` / aggregation / "Show daily active users for the last 30 days" / `SELECT event_date, COUNT(DISTINCT user_id) AS dau FROM events WHERE event_date >= CURRENT_DATE - INTERVAL '30' DAY GROUP BY event_date ORDER BY event_date`
  - `c06_join` / join / "Join orders with customers and list top 10 spenders" / `SELECT c.customer_id, c.name, SUM(o.order_total) AS spend FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.name ORDER BY spend DESC LIMIT 10`
  - `c07_join` / join / "Which products have orders but no shipments?" / `SELECT p.product_id FROM products p JOIN orders o ON o.product_id = p.product_id LEFT JOIN shipments s ON s.order_id = o.order_id WHERE s.order_id IS NULL`
  - `c08_timerange` / timerange / "What were sales between 2024-01-01 and 2024-03-31?" / `SELECT SUM(order_total) AS sales FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date <= DATE '2024-03-31'`
  - `c09_timerange` / timerange / "Compare this month's revenue to last month" / `SELECT date_trunc('month', order_date) AS month, SUM(revenue) AS revenue FROM orders WHERE order_date >= date_trunc('month', CURRENT_DATE) - INTERVAL '1' MONTH GROUP BY 1 ORDER BY 1`
  - `c10_anomaly` / anomaly / "Detect anomalies in daily signup counts" / `SELECT signup_date, COUNT(*) AS signups FROM users GROUP BY signup_date ORDER BY signup_date`
  - `c11_visualization` / visualization / "Plot the revenue trend for 2024" / `SELECT date_trunc('month', order_date) AS month, SUM(revenue) AS revenue FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01' GROUP BY 1 ORDER BY 1`
  - `c12_text2sql` / text2sql / "What columns describe customer churn?" / `` (empty: knowledge-only, `expected_tool_trajectory: ["search_knowledge"]`, `expected_result_contains: ["churn"]`)
  - `c13_text2sql` / text2sql / "Explain the orders fact table" / `` (empty: schema-only, `expected_tool_trajectory: ["show_schema"]`, `expected_result_contains: ["orders"]`)
  - `c14_text2sql` / text2sql / "What does the metric DAU mean?" / `` (empty: knowledge-only, `expected_tool_trajectory: ["search_knowledge"]`, `expected_result_contains: ["DAU"]`)
  - `c15_text2sql` / text2sql / "List the schema of the customers table" / `` (empty: schema-only, `expected_tool_trajectory: ["show_schema"]`, `expected_result_contains: ["customers"]`)
  - `c16_aggregation` / aggregation / "How many active users signed up last week?" / `SELECT COUNT(DISTINCT user_id) AS active_users FROM users WHERE signup_date >= CURRENT_DATE - INTERVAL '7' DAY`
  - `c17_visualization` / visualization / "Forecast next quarter revenue from history" / `SELECT date_trunc('month', order_date) AS month, SUM(revenue) AS revenue FROM orders GROUP BY 1 ORDER BY 1`
  - `c18_aggregation` / aggregation / "Break down conversion rate by channel" / `SELECT channel, SUM(conversions) * 1.0 / SUM(visits) AS conversion_rate FROM funnel GROUP BY channel`
  - `c19_report` / report / "Generate a sales report for Q1 2024" / `SELECT date_trunc('month', order_date) AS month, SUM(order_total) AS sales FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date <= DATE '2024-03-31' GROUP BY 1 ORDER BY 1` (`expected_tool_trajectory: ["search_knowledge", "text2sql", "save_report"]`, `expected_result_contains: ["report"]`)
  - `c20_report` / report / "Summarize order volume and save it as a report" / `SELECT date_trunc('day', order_date) AS day, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1` (`expected_tool_trajectory: ["text2sql", "save_report"]`, `expected_result_contains: ["report"]`)
  For each, set `expected_tool_trajectory` to match `_TRAJECTORIES` (drop the trailing `None`), and `expected_result_contains` to a lowercase substring of the final reply or a SQL keyword.

- [ ] **Step 9: Hand-record 19 cassettes matching each trajectory.**
  Each cassette is one JSONL line per tool call (the `tool`/`args`/`ok`/`result` shape from `t1.jsonl`), in trajectory order. The driver emits a `text2sql` proxy result of `{"sql": ..., "data": ...}`. Example `evals/runledger/cassettes/c02_aggregation.jsonl` (single line):
  ```json
  {"tool":"text2sql","args":{"reasoning":"Generate SQL","context":"How many orders were placed in 2024?"},"ok":true,"result":{"sql":"SELECT COUNT(*) FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'","data":"count\n42"}}
  ```
  Example `evals/runledger/cassettes/c19_report.jsonl` (three lines, one per tool in the report trajectory):
  ```json
  {"tool":"search_knowledge","args":{"reasoning":"Look up relevant knowledge","query_list":["Generate a sales report for Q1 2024"],"knowledge_bases":["columns"],"with_table_list":false},"ok":true,"result":{"columns":"# Relevant Columns\n- order_total\n- order_date"}}
  {"tool":"text2sql","args":{"reasoning":"Generate SQL","context":"Generate a sales report for Q1 2024"},"ok":true,"result":{"sql":"SELECT date_trunc('month', order_date) AS month, SUM(order_total) AS sales FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date <= DATE '2024-03-31' GROUP BY 1 ORDER BY 1","data":"month,sales\n2024-01,1000"}}
  {"tool":"save_report","args":{"content":"Report for: Generate a sales report for Q1 2024","title":"Generate a sales report for Q1 2","file_format":"md"},"ok":true,"result":"Saved report to ./data/q1_2024_sales_report.md"}
  ```
  The `args` in each cassette line must byte-match what `_scripted_llm_call`'s `_TOOL_ARGS_BUILDERS` emit for that prompt (RunLedger replay matches recorded calls by tool name + args). Author the remaining 17 the same way: one line per non-`None` trajectory entry, args built by the same builder, plausible deterministic `result`.

- [ ] **Step 10: Add the `eval` optional extra to `pyproject.toml`.**
  In `pyproject.toml`, under `[project.optional-dependencies]`, add an `eval` extra after the `test` block (which ends at L106):
  ```toml
  eval = [
      "runledger>=0.1.1,<0.2.0",
      "pyyaml>=6.0,<7.0",
  ]
  ```

- [ ] **Step 11: Add a smoke test asserting all cases load and trajectories are self-consistent.**
  Append to `tests/eval/test_runledger_agent.py`:
  ```python
  import glob
  import os

  import yaml

  _CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "runledger", "cases")


  def test_corpus_has_at_least_20_cases():
      files = glob.glob(os.path.join(_CASES_DIR, "*.yaml"))
      assert len(files) >= 20


  def test_every_case_has_category_and_gold():
      for path in glob.glob(os.path.join(_CASES_DIR, "*.yaml")):
          if os.path.basename(path) == "t1.yaml":
              continue  # seed case predates the gold schema
          with open(path) as fh:
              case = yaml.safe_load(fh)
          assert "category" in case, path
          assert "gold" in case, path
          assert "expected_tool_trajectory" in case["gold"], path


  def test_gold_trajectory_matches_driver_table():
      agent = _load_agent()
      for path in glob.glob(os.path.join(_CASES_DIR, "*.yaml")):
          if os.path.basename(path) == "t1.yaml":
              continue
          with open(path) as fh:
              case = yaml.safe_load(fh)
          prompt = case["input"]["prompt"]
          driver_traj = [t for t in agent._TRAJECTORIES[prompt] if t is not None]
          assert driver_traj == case["gold"]["expected_tool_trajectory"], path
  ```

- [ ] **Step 12: Run the eval test suite — expect pass.**
  Run: `uv run pytest tests/eval/test_runledger_agent.py -v`
  Expected: PASS — 8 tests green; ≥20 cases discovered, every case has `category`+`gold`, and each gold trajectory matches `_TRAJECTORIES`.

- [ ] **Step 13: Document the case/cassette/protocol format in the README.**
  Append a new section to `evals/runledger/README.md`:
  ```markdown
  ## Case / cassette / protocol format

  ### Case YAML (`cases/*.yaml`)
  - `id`, `description`, `input.prompt`, `cassette` — consumed by RunLedger.
  - `category` — one of `aggregation | join | timerange | anomaly | visualization | text2sql | report`.
  - `gold` — consumed by the out-of-band LLM judge (`evals/judge/`), ignored by RunLedger:
    - `expected_sql` — hand-written gold SQL (empty string for knowledge/schema-only cases).
    - `expected_tool_trajectory` — ordered list of tool names the agent should call.
    - `expected_result_contains` — lowercase substrings expected in the final reply.

  ### Cassette JSONL (`cassettes/*.jsonl`)
  One line per tool call, in trajectory order. Shape:
  `{"tool": <name>, "args": {...}, "ok": true, "result": <any>}`.
  The `args` must byte-match what the agent emits, because RunLedger replay matches
  recorded calls by tool name + args.

  ### Protocol note: keying on prompt
  The JSONL protocol carries no case-id. The agent driver (`agent/agent.py:_scripted_llm_call`)
  keys the scripted trajectory on the **user prompt text** (`_TRAJECTORIES`) and advances by the
  count of `ToolMessage`s already in history. Adding a case therefore requires (1) a `cases/*.yaml`,
  (2) a matching `cassettes/*.jsonl`, and (3) a `_TRAJECTORIES[prompt]` entry.
  ```

- [ ] **Step 14: Commit.**
  Run: `git add evals/runledger/cases evals/runledger/cassettes evals/runledger/agent/agent.py evals/runledger/suite.yaml evals/runledger/tools.py evals/runledger/README.md pyproject.toml tests/eval && git commit -m "Expand RunLedger corpus to 20+ cases with per-prompt scripted trajectories

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 17: LLM-as-Judge evaluator (wraps S2) + run_judge CLI + nightly judge CI job

**Files:**
- Create: `evals/judge/__init__.py`
- Create: `evals/judge/llm_judge.py` (`JudgeVerdict`, `LLMAsJudgeEvaluator` — wraps S2 `SimpleSQLEvaluator` from `openchatbi/text2sql/confidence.py`)
- Create: `evals/judge/run_judge.py` (`python -m evals.judge.run_judge --cases ... --out judge_out/report.json`, per-category aggregation)
- Create: `tests/eval/test_llm_judge.py`
- Modify: `.github/workflows/runledger.yml` (add a non-gating nightly `judge` job: `schedule` + `workflow_dispatch`, LLM key secret)

- [ ] **Step 1: Write a failing test for `LLMAsJudgeEvaluator.judge`.**
  The judge wraps S2's `SimpleSQLEvaluator` (`openchatbi/text2sql/confidence.py`, contract: `evaluate(question, sql, schema_info, data_sample) -> ConfidenceResult(score, reasons, checks)`). The verdict passes when the underlying score clears the threshold. Create `tests/eval/test_llm_judge.py`:
  ```python
  """Unit tests for the out-of-band LLM-as-Judge evaluator."""

  from unittest.mock import MagicMock

  from evals.judge.llm_judge import JudgeVerdict, LLMAsJudgeEvaluator


  def _evaluator_with_score(score, checks=None, reasons=None):
      from openchatbi.text2sql.confidence import ConfidenceResult

      inner = MagicMock()
      inner.evaluate.return_value = ConfidenceResult(
          score=score,
          reasons=reasons or ["looks correct"],
          checks=checks or {"select_columns": True, "where": True},
      )
      return LLMAsJudgeEvaluator(evaluator=inner, threshold=0.7)


  def test_judge_passes_when_score_above_threshold():
      judge = _evaluator_with_score(0.9)
      verdict = judge.judge(
          question="How many orders?",
          generated_sql="SELECT COUNT(*) FROM orders",
          expected_sql="SELECT COUNT(*) FROM orders",
      )
      assert isinstance(verdict, JudgeVerdict)
      assert verdict.passed is True
      assert verdict.score == 0.9
      assert verdict.dimensions == {"select_columns": True, "where": True}


  def test_judge_fails_when_score_below_threshold():
      judge = _evaluator_with_score(0.3)
      verdict = judge.judge(question="q", generated_sql="SELECT 1")
      assert verdict.passed is False
      assert "looks correct" in verdict.reasoning


  def test_judge_passes_through_schema_and_expected_sql():
      inner = MagicMock()
      from openchatbi.text2sql.confidence import ConfidenceResult

      inner.evaluate.return_value = ConfidenceResult(score=0.8, reasons=[], checks={})
      judge = LLMAsJudgeEvaluator(evaluator=inner, threshold=0.7)
      judge.judge(question="q", generated_sql="SELECT 1", expected_sql="SELECT 2", schema={"t": ["c"]})
      _, kwargs = inner.evaluate.call_args
      assert kwargs["question"] == "q"
      assert kwargs["sql"] == "SELECT 1"
      assert kwargs["schema_info"] == {"t": ["c"]}
      # expected_sql is folded into the data_sample context the inner evaluator sees
      assert "SELECT 2" in (kwargs["data_sample"] or "")
  ```

- [ ] **Step 2: Run the test — expect failure (module missing).**
  Run: `uv run pytest tests/eval/test_llm_judge.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'evals.judge.llm_judge'`.

- [ ] **Step 3: Implement `evals/judge/__init__.py` and `llm_judge.py`.**
  Create `evals/judge/__init__.py`:
  ```python
  """Out-of-band LLM-as-Judge evaluation (runs outside RunLedger)."""
  ```
  Create `evals/judge/llm_judge.py` (wraps S2; `threshold` is adjustable per §S2 "阈值需可调"):
  ```python
  from __future__ import annotations

  from dataclasses import dataclass, field

  from openchatbi.text2sql.confidence import SimpleSQLEvaluator


  @dataclass
  class JudgeVerdict:
      score: float
      passed: bool
      reasoning: str
      dimensions: dict[str, bool] = field(default_factory=dict)


  class LLMAsJudgeEvaluator:
      """Wraps the S2 SimpleSQLEvaluator to produce pass/fail verdicts for eval.

      Runs OUTSIDE RunLedger (custom assertion types are unsupported in runledger 0.1.1).
      """

      def __init__(self, evaluator: SimpleSQLEvaluator | None = None, threshold: float = 0.7) -> None:
          self._evaluator = evaluator or SimpleSQLEvaluator()
          self._threshold = threshold

      def judge(
          self,
          question: str,
          generated_sql: str,
          expected_sql: str | None = None,
          schema: dict | None = None,
      ) -> JudgeVerdict:
          # Fold the gold SQL into the data_sample context so the rubric can compare.
          data_sample = None
          if expected_sql:
              data_sample = f"Reference (gold) SQL for comparison:\n{expected_sql}"
          result = self._evaluator.evaluate(
              question=question,
              sql=generated_sql,
              schema_info=schema or {},
              data_sample=data_sample,
          )
          return JudgeVerdict(
              score=result.score,
              passed=result.score >= self._threshold,
              reasoning="; ".join(result.reasons),
              dimensions=result.checks,
          )
  ```

- [ ] **Step 4: Run the test — expect pass.**
  Run: `uv run pytest tests/eval/test_llm_judge.py -v`
  Expected: PASS — 3 tests green; verdict thresholds on the wrapped S2 score and forwards schema/expected_sql.

- [ ] **Step 5: Write a failing test for `run_judge` per-category aggregation.**
  The runner loads `cases/*.yaml`, reads each `gold` block, judges the gold SQL, and aggregates pass-rate + mean score per `category`. Add to `tests/eval/test_llm_judge.py`:
  ```python
  import json
  import os

  from evals.judge import run_judge


  def _write_case(tmp_path, name, category, sql):
      p = tmp_path / f"{name}.yaml"
      p.write_text(
          "id: %s\n"
          "category: %s\n"
          "input:\n"
          "  prompt: 'q for %s'\n"
          "gold:\n"
          "  expected_sql: \"%s\"\n"
          "  expected_tool_trajectory: ['text2sql']\n"
          "  expected_result_contains: ['x']\n" % (name, category, name, sql)
      )
      return p


  def test_run_judge_aggregates_per_category(tmp_path, monkeypatch):
      cases_dir = tmp_path / "cases"
      cases_dir.mkdir()
      _write_case(cases_dir, "a1", "aggregation", "SELECT COUNT(*) FROM orders")
      _write_case(cases_dir, "j1", "join", "SELECT * FROM a JOIN b ON a.id=b.id")

      # Stub the judge so the test is hermetic (no LLM).
      from evals.judge.llm_judge import JudgeVerdict

      def fake_judge_factory():
          calls = {"n": 0}

          class _Stub:
              def judge(self, question, generated_sql, expected_sql=None, schema=None):
                  calls["n"] += 1
                  # aggregation passes, join fails
                  passed = "JOIN" not in generated_sql
                  return JudgeVerdict(
                      score=0.9 if passed else 0.3,
                      passed=passed,
                      reasoning="stub",
                      dimensions={},
                  )

          return _Stub()

      monkeypatch.setattr(run_judge, "_build_judge", lambda: fake_judge_factory())
      out_path = tmp_path / "judge_out" / "report.json"
      exit_code = run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path))
      assert exit_code == 0
      report = json.loads(out_path.read_text())
      assert report["by_category"]["aggregation"]["pass_rate"] == 1.0
      assert report["by_category"]["join"]["pass_rate"] == 0.0
      assert report["overall"]["total"] == 2
      assert os.path.exists(out_path)
  ```

- [ ] **Step 6: Run the test — expect failure (runner missing).**
  Run: `uv run pytest tests/eval/test_llm_judge.py -v`
  Expected: FAIL — `AttributeError: module 'evals.judge.run_judge' has no attribute 'run'`.

- [ ] **Step 7: Implement `evals/judge/run_judge.py`.**
  Create `evals/judge/run_judge.py` (CLI + per-category aggregation; runs OUTSIDE RunLedger). Cases with empty `expected_sql` (knowledge/schema-only) are skipped since there is no SQL to judge:
  ```python
  from __future__ import annotations

  import argparse
  import glob
  import json
  import os
  import statistics
  import sys
  from typing import Any

  import yaml

  from evals.judge.llm_judge import LLMAsJudgeEvaluator


  def _build_judge() -> LLMAsJudgeEvaluator:
      # Real LLM judge; the threshold is config-adjustable per S2.
      return LLMAsJudgeEvaluator(threshold=0.7)


  def _load_cases(cases_dir: str) -> list[dict[str, Any]]:
      cases = []
      for path in sorted(glob.glob(os.path.join(cases_dir, "*.yaml"))):
          with open(path) as fh:
              case = yaml.safe_load(fh) or {}
          if "gold" in case and case["gold"].get("expected_sql"):
              cases.append(case)
      return cases


  def run(cases_dir: str, out_path: str) -> int:
      judge = _build_judge()
      cases = _load_cases(cases_dir)
      results: list[dict[str, Any]] = []
      for case in cases:
          gold = case["gold"]
          verdict = judge.judge(
              question=case["input"]["prompt"],
              generated_sql=gold["expected_sql"],
              expected_sql=gold["expected_sql"],
          )
          results.append(
              {
                  "id": case["id"],
                  "category": case.get("category", "uncategorized"),
                  "score": verdict.score,
                  "passed": verdict.passed,
                  "reasoning": verdict.reasoning,
              }
          )

      by_category: dict[str, dict[str, Any]] = {}
      for r in results:
          bucket = by_category.setdefault(r["category"], {"scores": [], "passed": 0, "total": 0})
          bucket["scores"].append(r["score"])
          bucket["passed"] += 1 if r["passed"] else 0
          bucket["total"] += 1
      for cat, bucket in by_category.items():
          scores = bucket.pop("scores")
          bucket["mean_score"] = statistics.mean(scores) if scores else 0.0
          bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0

      overall_passed = sum(1 for r in results if r["passed"])
      report = {
          "overall": {
              "total": len(results),
              "passed": overall_passed,
              "pass_rate": (overall_passed / len(results)) if results else 0.0,
              "mean_score": statistics.mean([r["score"] for r in results]) if results else 0.0,
          },
          "by_category": by_category,
          "cases": results,
      }
      os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
      with open(out_path, "w") as fh:
          json.dump(report, fh, indent=2)
      return 0


  def main(argv: list[str] | None = None) -> int:
      parser = argparse.ArgumentParser(prog="evals.judge.run_judge")
      parser.add_argument("--cases", default="evals/runledger/cases")
      parser.add_argument("--out", default="judge_out/report.json")
      args = parser.parse_args(argv)
      return run(cases_dir=args.cases, out_path=args.out)


  if __name__ == "__main__":
      raise SystemExit(main())
  ```

- [ ] **Step 8: Run the test — expect pass.**
  Run: `uv run pytest tests/eval/test_llm_judge.py -v`
  Expected: PASS — 4 tests green; report aggregates pass-rate + mean score per category and writes `report.json`.

- [ ] **Step 9: Add a non-gating nightly judge job to the CI workflow.**
  In `.github/workflows/runledger.yml`, add `schedule` to the `on:` block (L2-6) and append a separate `judge` job after the existing `runledger` job (the judge job is independent and never gates PRs):
  ```yaml
  on:
    workflow_dispatch:
    schedule:
      - cron: "0 7 * * *"  # nightly 07:00 UTC, non-gating judge run
    pull_request:
      paths:
        - "openchatbi/**"
  ```
  Append after the `runledger` job:
  ```yaml
    judge:
      # Non-gating LLM-as-Judge run (nightly + manual). Never blocks PR merges:
      # only runs on schedule or explicit dispatch, and uploads a report artifact.
      if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
      runs-on: ubuntu-latest
      continue-on-error: true
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            python -m pip install ".[eval]"
        - name: Run LLM-as-Judge over gold corpus
          env:
            OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          run: |
            python -m evals.judge.run_judge --cases evals/runledger/cases --out judge_out/report.json
        - name: Upload judge report
          uses: actions/upload-artifact@v4
          with:
            name: judge-report
            path: judge_out/**
  ```

- [ ] **Step 10: Run the full eval test suite — expect pass.**
  Run: `uv run pytest tests/eval/ -v`
  Expected: PASS — Task 16 driver/corpus tests and Task 17 judge/runner tests all green.

- [ ] **Step 11: Commit.**
  Run: `git add evals/judge tests/eval/test_llm_judge.py .github/workflows/runledger.yml && git commit -m "Add out-of-band LLM-as-Judge evaluator, run_judge CLI, and nightly judge CI job

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---
