## Context

OpenChatBI 的 README 中提及支持“多维度异常下钻根因分析”，但当前仅有 `adtributor_demo.py` 作为概念验证实现。在真实的 BI 问答场景中，当指标发生异常波动（如流量下降）时，Agent 需要一种系统化的方法来判断是哪个维度的哪些元素（如“平台=iOS”或“渠道=自然搜索”）导致了整体异常。

同时，原版 `adtributor_demo.py` 的输入契约为 `dict[str, DataFrame]`（每个维度独立一张一维聚合表）。如果直接暴露给 Agent，会让 Agent 难以构建合适的数据结构（极易引发维度爆炸或让 Agent 多次书写复杂的聚合 SQL）。另外，原版输出的 `Surprise` 和 `Explanatory Power (EP)` 对最终用户不够友好，缺乏业务可解释性。

## Goals / Non-Goals

**Goals:**
1. 将微软 Adtributor 算法的 Demo 重构为独立、可测试的核心算法库，支持绝对指标和派生（比率）指标的归因。
2. 设计分层架构：底层维持高性能的 1D 聚合运算，顶层（LangChain Tool）封装为面向 Agent 友好的“一维长表（Melted Table）”输入契约。
3. Tool 层负责将算法输出的专业指标转化为自然语言业务解释（Narrative），提升 Agent 回答的质量。

**Non-Goals:**
1. 自动生成 SQL 拉取包含同比/环比数据的基准值（Baseline）。该逻辑由 Agent 在 Text2SQL 阶段或调用前预处理完成。
2. 发现异常发生的时间点。本变更关注的是**发生异常后的维度下钻归因**，而不是时序异常检测本身。
3. 支持层级维度（Hierarchy）的自动剪枝。初版仅支持平行的正交维度归因。

## Decisions

### Decision 1: 分层的数据结构契约

**方案**：算法层保留 `dict[str, DataFrame]` 结构；Tool 包装层采用单一“一维长表 (Melted Table)”结构（`[{"dimension_name": "x", "predict": 100, ...}]`）。
**替代方案考虑**：
- **方案 A（直接使用宽表）**：传入多维度明细宽表。**拒绝原因**：在高基数维度下会引发笛卡尔积爆炸（Curse of Dimensionality），导致 OOM 及 Token 超载。
- **方案 B（让 Agent 直接构造 Dict）**：**拒绝原因**：Agent 对 Python 嵌套字典和多个 DataFrame 实例的组装能力较弱，容易产生语法错误。
**Rationale**：
一维长表的记录数等于各维度基数之和，完美规避了维度爆炸。同时，二维的长表格式是 Text2SQL 最自然的输出形态（SQL 的 `UNION ALL` 或工具预处理后即可得），大大降低了 Agent 的使用心智门槛。

### Decision 2: 业务逻辑与算法分离

**方案**：Adtributor 核心算法不处理时间窗对比（如“今日 vs 昨日”），仅接收抽象的 `real`（实际值）与 `predict`（预测值/基准值）。
**Rationale**：
算法需要保持纯粹。将“什么是 predict（可能是昨天，也可能是上周，甚至是模型预测值）”的定义权完全交给上层的业务调用方（Agent/用户）。这保证了算法的泛化能力。

### Decision 3: 自动生成业务解释（Narrative）

**方案**：在 Tool 层将 Adtributor 的 `EP`（解释力）翻译为 `Contribution`（贡献度），并在 `DimensionResult` 中直接输出预生成的中文自然语言结论（如“某维度的某元素贡献了 85% 的波动”）。
**Rationale**：
如果仅返回 `{"explanatory_power": 0.85}`，LLM 可能会产生错觉甚至拒绝解释。Tool 层面生成 Narrative 可以作为一个可靠的 Fallback 或直接被 Agent 引用，保证最终输出的稳定性。

## Risks / Trade-offs

- **[Risk]** Agent 无法写出构造一维长表的 SQL。
  → **Mitigation**：在 `agent_prompt.md` 或 Tool 的 description 中，提供明确的 Prompt 示例，指导 Agent 首先通过 Text2SQL 查询明细，或者通过多次独立查询并在 Tool 调用参数中用 JSON 组装（Tool 接受 `list[dict]` 格式的 JSON）。

- **[Risk]** 派生指标的分子分母数据难以获取。
  → **Mitigation**：允许回退。如果难以获取分子分母，用户可降级使用绝对指标模式（`derived=false`）分析比率本身（虽不符合严格数学推导，但能做基础判断）。Tool 参数中 `derived` 作为显式开关。

## Migration Plan

1. 创建新的 `openchatbi/analysis/adtributor.py` 和 `openchatbi/analysis/models.py`。
2. 创建 `openchatbi/tool/adtributor_tool.py`，注册到 `mcp_tools` 或 `agent_graph.py` 的可用工具中。
3. 补充单元测试。
4. 验证通过后，将原 `openchatbi/prompts/analysis/adtributor_demo.py` 标记为废弃（Deprecated）或删除。