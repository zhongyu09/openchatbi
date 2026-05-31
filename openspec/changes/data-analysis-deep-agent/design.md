## Context

OpenChatBI 主 Agent（`agent_graph.py`）采用 LangGraph `StateGraph` + `ToolNode` 模式，已将 Text2SQL 子图、时序预测、Python 执行等注册为同级工具。复杂数据分析需多步编排（取数 → 检测/预测 → 下钻 → 关联分析 → 总结），平铺工具模式导致主 Agent 上下文膨胀、步骤遗漏。

[Deep Agents](https://github.com/langchain-ai/deepagents) 在 LangGraph 之上提供子 Agent 委托、上下文管理与长时程任务编排。项目已升级至 `langgraph==1.1.10`，具备集成条件。

两个并行变更负责核心工具实现，且均将工具放置在 `openchatbi/tool/` 包下：

| 并行变更 | 产出工具 | 本变更职责 |
|----------|----------|-----------|
| `anomaly-detection-algorithm` | `openchatbi/tool/anomaly_detection.py` | 直接 import 注册到数据分析 Agent |
| `adtributor-anomaly-drilldown` | `openchatbi/tool/adtributor_tool.py` | 直接 import 注册到数据分析 Agent |

## Goals / Non-Goals

**Goals:**

- 构建专用数据分析 Agent，支持 5 类分析场景
- 通过 `data_analysis` 工具将分析任务委托给子 Agent
- 直接复用 `openchatbi/tool/` 下的 `anomaly_detection` 与 `adtributor_drilldown` 工具
- 复用现有 text2sql / forecast / python 工具
- 支持 sync/async 两种 graph 构建模式

**Non-Goals:**

- 不重复实现异常检测算法或其工具封装（由并行变更负责）
- 不重复实现 Adtributor 核心或其工具封装（由并行变更负责）
- 不替换或重构主 Agent 图结构
- 不改造前端 UI

## Decisions

### 1. 集成模式：数据分析 Agent 作为子 Agent + 主 Agent 委托工具

**选择**：`create_deep_agent` 构建数据分析 Agent；主 Agent 注册 `data_analysis` StructuredTool。

### 2. 工具注入策略

| 工具 | 来源 | 本变更范围 |
|------|------|-----------|
| `text2sql` | `get_sql_tools(sql_graph, sync_mode)` | 复用 |
| `timeseries_forecast` | `openchatbi/tool/timeseries_forecast.py` | 复用 |
| `run_python_code` | `openchatbi/tool/run_python_code.py` | 复用 |
| `anomaly_detection` | `openchatbi/tool/anomaly_detection.py` | **直接 import**（来自并行变更） |
| `adtributor_drilldown` | `openchatbi/tool/adtributor_tool.py` | **直接 import**（来自并行变更） |

### 3. anomaly_detection 与 adtributor_drilldown 的复用

**选择**：数据分析 Agent 直接 import 并行变更产出的现成工具，不创建任何 wrapper：

```python
from openchatbi.tool.anomaly_detection import anomaly_detection
from openchatbi.tool.adtributor_tool import adtributor_drilldown
```

**I/O 契约**：完全遵循并行变更中定义的 Tool 层契约。
- `anomaly_detection`：依赖 `timeseries_forecast` 服务，多因素打分。
- `adtributor_drilldown`：接收一维长表数据（`data: list[dict]`），返回带业务解释（Narrative）的根因分析结果。

**健康检查**：`anomaly_detection` 与 `timeseries_forecast` 共享 forecast 服务健康检查（服务不可用时两者均排除）。

### 4. 分析场景工作流（Prompt 驱动）

| 场景 | 典型流程 |
|------|----------|
| 单指标趋势预测 | text2sql → timeseries_forecast → 解读 |
| 单指标异常检测 | text2sql → anomaly_detection → 解读 |
| 单指标异常维度下钻 | text2sql 取一维长表明细 → adtributor_drilldown → 解读 |
| 多指标关联 | text2sql → run_python_code → 解读 |
| 业务组合分析 | text2sql → run_python_code → 解读 |

### 5. 模块结构

实际落地的分层为「算法核心在 `analysis/`，Tool 封装在 `tool/`，数据分析 Agent 编排在 `analysis/agent.py`」：

```
openchatbi/analysis/
├── __init__.py
├── agent.py                    # build_data_analysis_agent(), get_data_analysis_tool()
│                               #   + _build_sub_agent_config() / _extract_final_content() 辅助函数
├── anomaly_detection.py        # 异常检测算法核心（来自 anomaly-detection-algorithm）
├── adtributor.py               # Adtributor 算法核心（来自 adtributor-anomaly-drilldown）
└── models.py                   # Adtributor 输出数据模型（AdtributorOutput 等）

openchatbi/tool/
├── anomaly_detection.py        # anomaly_detection Tool 封装（调用 analysis.anomaly_detection）
└── adtributor_tool.py          # adtributor_drilldown Tool 封装（调用 analysis.adtributor）

openchatbi/prompts/
└── data_analysis_prompt.md
```

> 注：早期的概念验证文件 `analysis/adtributor_demo.py` 已删除，由 `analysis/adtributor.py` + `tool/adtributor_tool.py` 取代。

### 6. 主 Agent 集成点

```python
data_analysis_tool = get_data_analysis_tool(
    sql_graph=sql_graph,
    sync_mode=sync_mode,
    llm_provider=llm_provider,
    checkpointer=checkpointer,
    memory_store=memory_store,
)
normal_tools.append(data_analysis_tool)
```

## Risks / Trade-offs

- **[Risk] 并行变更未合并导致工具不可用** → apply 前先完成或 cherry-pick `anomaly-detection-algorithm` 与 `adtributor-anomaly-drilldown`
- **[Risk] deepagents 版本冲突** → pin 兼容版本，CI 验证
- **[Trade-off] 三个变更需协调合并顺序** → 建议先 merge 两个工具变更，再 apply 本变更

## Migration Plan

1. 确认 `openchatbi/tool/anomaly_detection.py` 与 `openchatbi/tool/adtributor_tool.py` 可用
2. 添加 `deepagents` 依赖
3. 实现 analysis agent
4. 集成到 `agent_graph.py`
5. 回滚：移除 `data_analysis` 工具注册

## Resolved Decisions（原 Open Questions）

- **合并顺序**：anomaly-detection-algorithm / adtributor-anomaly-drilldown → data-analysis-deep-agent（两个工具变更先合并）。
- **独立 LLM provider**：已支持。新增可选的 `analysis_llm` 配置项（`Config` / `LLMProviderConfig` 均含该字段），并由 `openchatbi.llm.llm.get_analysis_llm()` 解析；未配置时回退到 `default_llm`。数据分析 Agent 通过 `get_analysis_llm(llm_provider)` 获取模型。

## 子 Agent 配置隔离（checkpointer / thread_id）

数据分析 Agent 是独立编译的子图，可能与主 Agent 共享同一 `checkpointer`。`get_data_analysis_tool` 的 sync/async 调用函数通过注入的 `RunnableConfig` 派生子配置（`_build_sub_agent_config`）：

- 由父 `thread_id` 派生确定性子线程 id：`"{parent_thread_id}:data_analysis"`，既满足 checkpointer 对 `thread_id` 的要求，又避免污染父图 checkpoint 线程，同时保证 `GraphInterrupt` 恢复时 id 稳定。
- 清除继承的 `checkpoint_ns` / `checkpoint_id`，让子 Agent 从干净命名空间开始。
- 其余 config（callbacks/tags/metadata）原样透传。
- 返回值经 `_extract_final_content` 兜底：content 为多模态 content blocks（list）时拼接文本，避免返回非 str。
