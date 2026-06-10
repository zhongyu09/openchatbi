# Judge Eval：端到端 Text2SQL 评测

这个目录提供一套**端到端 judge 评测**：给定一组带 GOLD SQL 的用例，先让真实 agent 针对每个问题生成 SQL，再用 LLM judge 把「生成 SQL」和「GOLD SQL」逐条打分。

`example_cases/` 里是针对自带 demo 库（`example/tracking_orders.sqlite`，dialect=sqlite）写好的 10 个 schema-matched GOLD 用例，可直接跑通。

## 1. 两步走流程

整套评测分两步：先 `collect_generated` 收集 agent 生成的 SQL，再 `run_judge` 打分。

```bash
# 第一步：跑真实 agent，把每个用例生成的 SQL 收集成一个 map 文件
python -m evals.judge.collect_generated \
  --cases evals/judge/example_cases \
  --config example/config.yaml \
  --out generated.json

# 第二步：用 LLM judge 把 generated 与 GOLD 逐条对比打分，输出报告
python -m evals.judge.run_judge \
  --cases evals/judge/example_cases \
  --generated generated.json \
  --out judge_out/report.json
```

注意：

- **两步都需要真实 LLM key**。`collect_generated` 要真正驱动 agent graph 生成 SQL，`run_judge` 用的是真实 LLM judge（`LLMAsJudgeEvaluator`，内部复用 `SimpleSQLEvaluator` 打分）。请把 `example/config.yaml` 里 `default_llm`（以及 `embedding_model`）的 `api_key: YOUR_API_KEY_HERE` 换成你的真实 key（或改用文件里注释好的 Ollama 本地 LLM）。
- `collect_generated --out` 默认 `generated.json`，默认 `--format json`，产出 `{"<case_id>": "SELECT ...", ...}` 的 id→SQL map；也可以 `--format jsonl` 产出每行 `{"id","prompt","generated_sql"}`。两种格式 `run_judge --generated` 都能直接消费。
- `run_judge` 通过 `case_id` 优先、`prompt` 兜底来匹配生成的 SQL；匹配不到的用例会被标记 `skipped` 且不计入分数。
- 生成的 SQL 是从 checkpointer 终态读取的（`graph.get_state(config).values["sql"]`，与 `run_cli` 一致）——agent graph 用 `output_schema=OutputState` 编译，`invoke()` 的返回值里不含 `sql`，必须读终态。agent 若停在 human-in-the-loop 中断（`snapshot.next` 非空），表示没有提交 SQL，该用例记为空 SQL。

报告 `judge_out/report.json` 含 overall（pass_rate / mean_score）、按 category 的聚合，以及每个用例的 score / passed / reasoning。

## 2. 用例文件 schema

每个用例是 `*.yaml`，放在一个 cases 目录下（如 `example_cases/`）。字段：

```yaml
id: ex01_simple_select          # 必填，唯一 id，命名约定 ex<NN>_<category>
category: simple_select         # 必填，用于报告里按类聚合
description: "一句话说明"         # 可选
input:
  prompt: "用户自然语言问题"        # 必填，agent 和 judge 都用它
gold:
  expected_sql: "SELECT ..."     # 必填，标准答案 SQL（本库为 sqlite 方言）
  expected_tool_trajectory: ["text2sql"]   # 可选，期望工具轨迹
  expected_result_contains: ["..."]        # 可选，期望结果包含的关键词
```

`run_judge._load_cases` 只会加载带 `gold.expected_sql` 的用例。`example_cases/` 覆盖了
simple_select / filter / aggregation / group_by / join / multi_join / subquery /
order_limit(top-N) / distinct / having 共 10 类，每条 GOLD SQL 都已对真实库验证过「能执行且至少返回一行」。

## 3. 换成你自己的库

1. **指向你的数据库**：把 `example/config.yaml` 里 `data_warehouse_config.uri` 改成你的库连接串（如 `postgresql://...`、`mysql://...`、或别的 sqlite 文件），并把 `dialect` 改成对应方言。首次运行时 catalog 会通过 `catalog/factory.py` 自动从库里加载 `table_info`。
2. **（可选）手写 catalog 元数据**：若不想自动反射，可在 catalog 的 `data_path`（demo 里是 `./example`）下手写 `table_info.yaml` 和 `sql_example.yaml`，agent 会优先用这些。
3. **写你自己的 GOLD 用例**：参照 `example_cases/` 的 schema，在你自己的 cases 目录下按你的真实表/列名写 GOLD 用例。**务必用你的库验证每条 `expected_sql` 能执行且返回行**，注意用你库对应的 SQL 方言（日期函数、引号等）。
4. **跑评测**：把上面两步命令的 `--cases` 指向你的 cases 目录、`--config` 指向你的 config，依次 `collect_generated` → `run_judge`。

## 4. Smoke 模式

`run_judge` 不传 `--generated` 时进入 smoke 模式：它用 GOLD SQL 同时当「生成」和「期望」喂给 judge（gold-vs-gold），仅用于验证整条评测链路接通，**分数没有实际意义**。
