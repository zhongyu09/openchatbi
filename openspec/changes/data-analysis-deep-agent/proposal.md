## Why

OpenChatBI 主 Agent 当前将 Text2SQL、时序预测、Python 执行等工具平铺在同一层级，缺乏面向复杂数据分析场景（趋势预测、异常检测、维度下钻、多指标关联、组合分析）的专用编排能力。用户提出「为什么 GMV 下降」「哪些渠道贡献了 CTR 异常」等问题时，主 Agent 需自行规划多步工具链，容易遗漏步骤或上下文膨胀。基于 [Deep Agents](https://github.com/langchain-ai/deepagents) 构建专用数据分析子 Agent，可隔离分析上下文、复用现有工具，并将复杂分析任务委托给具备领域 Prompt 与工具集的数据分析 Agent，从而补齐 README 承诺的高级分析能力。

## What Changes

- 新增 `openchatbi/analysis/` 模块，基于 `deepagents.create_deep_agent` 构建数据分析 Agent。
- 为数据分析 Agent 挂载工具：
  - `text2sql`（复用 `get_sql_tools` 包装）
  - `timeseries_forecast`（复用 `openchatbi/tool/timeseries_forecast.py`）
  - `run_python_code`（复用 `openchatbi/tool/run_python_code.py`）
  - `anomaly_detection`（**直接复用**并行变更 `anomaly-detection-algorithm` 产出的 `openchatbi/tool/anomaly_detection.py`）
  - `adtributor_drilldown`（**直接复用**并行变更 `adtributor-anomaly-drilldown` 产出的 `openchatbi/tool/adtributor_tool.py`）
- 在主 Agent（`agent_graph.py`）注册 `data_analysis` 工具，将分析类请求委托给数据分析 Agent 子图执行。
- 新增数据分析 Agent 系统 Prompt（`openchatbi/prompts/data_analysis_prompt.md`），覆盖 5 类分析场景的工作流指引。
- 更新主 Agent Prompt（`agent_prompt.md`），说明何时委托 `data_analysis` 工具。
- 新增 `deepagents` 依赖及单元/集成测试。

## Capabilities

### New Capabilities

- `data-analysis-agent`：基于 Deep Agents 的数据分析子 Agent，挂载到主 Agent，编排 text2sql / 预测 / 异常检测 / 下钻 / Python 分析，支持单指标趋势预测、单指标异常检测、单指标异常维度下钻、多指标关联、基于业务的组合分析。
- `anomaly-detection`：本变更**不重复实现**，直接复用并行变更产出的工具。
- `anomaly-drilldown`：本变更**不重复实现**，直接复用并行变更产出的 `adtributor_drilldown` 工具。

### Modified Capabilities

（无现有 OpenSpec 规格需修改）

## Impact

- **新增代码**：`openchatbi/analysis/agent.py`、`openchatbi/prompts/data_analysis_prompt.md`、`tests/analysis/test_data_analysis_agent.py`。
- **复用代码**：
  - `openchatbi/tool/anomaly_detection.py`（来自并行变更）
  - `openchatbi/tool/adtributor_tool.py`（来自并行变更）
  - `get_sql_tools`、`timeseries_forecast`、`run_python_code`
- **修改代码**：`openchatbi/agent_graph.py`（注册 `data_analysis` 委托工具）、`openchatbi/prompts/agent_prompt.md`、可选 `openchatbi/config.yaml.template`。
- **依赖**：新增 `deepagents`（基于 LangGraph，与现有 `langgraph==1.1.10` 兼容）。
- **非目标（本变更）**：异常检测或 Adtributor 核心算法实现；替换主 Agent 架构；前端 UI 改造。
