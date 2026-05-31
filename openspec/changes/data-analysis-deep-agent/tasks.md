## 1. 依赖与模块脚手架

- [x] 1.1 在 `pyproject.toml` 添加 `deepagents` 依赖，验证与 `langgraph==1.1.10` 兼容并刷新 lock 文件
- [x] 1.2 创建 `openchatbi/analysis/` 模块结构（`__init__.py`、`agent.py`）；确认并行变更产出可用：
  - `anomaly-detection-algorithm` → `openchatbi/tool/anomaly_detection.py`
  - `adtributor-anomaly-drilldown` → `openchatbi/tool/adtributor_tool.py`

## 2. 数据分析 Agent

- [x] 2.1 编写 `openchatbi/prompts/data_analysis_prompt.md`：5 类分析场景工作流与工具使用指引（强调下钻前使用 text2sql 准备一维长表数据）
- [x] 2.2 在 `openchatbi/analysis/agent.py` 中 import `anomaly_detection` 和 `adtributor_drilldown` 工具
- [x] 2.3 在 `openchatbi/analysis/agent.py` 实现 `build_data_analysis_agent()`：组装 text2sql、timeseries_forecast、run_python_code、anomaly_detection、adtributor_drilldown
- [x] 2.4 实现与 `timeseries_forecast` 一致的健康检查逻辑：forecast 服务不可用时排除 `timeseries_forecast` 和 `anomaly_detection`
- [x] 2.5 在 `openchatbi/analysis/agent.py` 实现 `get_data_analysis_tool()`：StructuredTool 包装，sync/async invoke，GraphInterrupt 传播
- [x] 2.6 编写 `tests/analysis/test_data_analysis_agent.py`：Agent 创建、工具集注册、mock 调用验证

## 3. 主 Agent 集成

- [x] 3.1 在 `openchatbi/agent_graph.py` 的 `_build_graph_core` 中注册 `data_analysis` 工具
- [x] 3.2 更新 `openchatbi/prompts/agent_prompt.md`：添加 `data_analysis` 委托策略
- [x] 3.3 编写 `tests/analysis/test_agent_graph_integration.py`：主 Agent graph 构建包含 `data_analysis` 工具

## 4. 配置与文档

- [x] 4.1 可选：在 `config.yaml.template` 添加 `analysis_llm` 配置项
- [x] 4.2 更新 `docs/source/tools.rst` 或 README：说明数据分析 Agent 及工具依赖关系

## 5. 验证

- [x] 5.1 运行 `pytest tests/analysis/` 确保全部通过
- [x] 5.2 运行现有 `tests/context_management/test_agent_graph_integration.py` 等回归测试，确认无破坏
