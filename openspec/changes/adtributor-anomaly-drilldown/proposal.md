## Why

OpenChatBI 在 README 中承诺提供「多维度异常下钻根因分析」能力，但目前仅有 `adtributor_demo.py` 参考实现，缺少正式模块、清晰的 I/O 契约以及与 Agent 工作流的集成。当用户发现指标异常（如 CTR 下降、GMV 突增）时，Agent 无法系统化地按维度归因并给出可解释的下钻结论。基于微软 Adtributor 算法实现标准化异常下钻分析，可补齐这一核心分析能力，并为后续 Agent Tool 封装和自动化报告提供基础。

## What Changes

- 新增 `openchatbi/analysis/adtributor.py` 核心算法模块，将 `adtributor_demo.py` 中的逻辑重构为可测试、可复用的库代码（去除对外部 `df_utils` 的依赖）。
- 定义清晰的 **输入/输出契约**（见下方），支持绝对指标与派生指标（比率类，如 CTR = clicks/impressions）两种模式。
- 支持 `drop`（指标下降）与 `rise`（指标上升）两类异常方向的下钻分析。
- 新增 LangChain Tool（`adtributor_drilldown`），供 Agent 在检测到异常后调用，输入各维度 baseline/actual 数据，输出结构化根因候选。
- 新增单元测试，覆盖 surprise、explanatory power 计算及典型下钻场景。
- 保留 `adtributor_demo.py` 作为历史参考，或在实现完成后标记为 deprecated 并指向新模块。

### 输入契约与架构分层

为了在“避免维度组合爆炸”与“降低 Agent 调用负担”之间取得平衡，同时剥离与核心算法无关的业务逻辑（如同比/环比的数据准备），我们采用**分层架构**：

1. **核心算法层 (`openchatbi/analysis/adtributor.py`)**
   - **职责**：纯粹的数学计算（Surprise、Explanatory Power）、候选过滤。完全不关心什么是“上周”、什么是“昨天”，只认 `real` 和 `predict`。
   - **输入契约**：保留原版的 `dict[str, DataFrame]`，即每个维度独立传入一个 1D 聚合表（避免高维宽表的笛卡尔积爆炸）。
2. **工具封装层 (`openchatbi/tool/adtributor_tool.py`)**
   - **职责**：面向 Agent 的友好接口。接收 Agent 生成的单一“一维长表 (Melted Table)”，负责在内部将其拆解为 `dict[str, DataFrame]` 喂给核心算法。生成带业务解释性的输出。
   - **数据准备**：Agent 调用 Text2SQL 时，由 Agent/SQL 负责计算好实际值 (real) 与基准值 (predict，如昨日同期)，工具层只负责读取处理好的长表结果。

**Tool 层输入契约（面向 Agent）：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `data` | `list[dict]` | 是 | 一维长表数据。包含维度名、维度值、实际值、基准值（见下文格式） |
| `derived` | `bool` | 是 | 是否为派生指标（比率类）。`false` 时使用绝对指标，`true` 时使用分子/分母 |
| `issue_type` | `"drop"` \| `"rise"` | 否 | 异常方向，默认 `"drop"` |

**Tool 传入数据的长表结构要求：**

绝对指标模式（`derived=false`）：
包含 `dimension_name`, `element_value`, `predict`, `real`，可选 `proportion`, `base_proportion`。

派生指标模式（`derived=true`）：
包含 `dimension_name`, `element_value`, `predict_numerator`, `predict_denominator`, `real_numerator`, `real_denominator`，可选 `proportion`, `base_proportion`。

**示例（Agent 传入的长表数据）：**
```python
input_data = [
    {"dimension_name": "site_section", "element_value": 1, "predict": 0.5, "real": 0.7},
    {"dimension_name": "site_section", "element_value": 2, "predict": 0.1, "real": 0.9},
    {"dimension_name": "device", "element_value": "ios", "predict": 1000, "real": 800},
    {"dimension_name": "device", "element_value": "android", "predict": 2000, "real": 2100}
]
```

### 输出契约（Tool 层输出）

Tool 的输出除了底层算法的原始指标外，重点增强了**业务解释（Narrative）**。

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `str` | 分析状态：`"success"` / `"no_root_cause"` / `"no_anomaly_direction"` |
| `root_causes` | `dict[str, list[Any]]` | 根因候选：维度名 → 该维度下贡献最大的元素列表 |
| `dimension_details` | `dict[str, DimensionResult]` | 每个维度的详细诊断信息与业务解释 |

**DimensionResult 结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `contribution` | `float` | 对整体异常的贡献度百分比（即算法层的累积 EP，如 `0.85`） |
| `elements` | `list[Any]` \| `null` | 根因元素；未通过则为 `null` |
| `narrative` | `str` | **自动生成的业务解释**，如："该维度的 [广东, 浙江] 元素贡献了 85.00% 的整体波动。" |
| `raw_metrics` | `dict` | 包含算法底层的 `total_surprise`, `surprise`, `reason` 等细节 |

## Capabilities

### New Capabilities

- `adtributor-drilldown`: 基于微软 Adtributor 算法的多维度异常下钻根因分析，包含核心算法库、输入/输出契约、Agent Tool 集成及单元测试。

### Modified Capabilities

（无现有 OpenSpec 规格需修改）

## Impact

- **新增代码**：`openchatbi/analysis/` 目录（`adtributor.py`、`models.py`、`tool.py`）、`tests/analysis/` 测试目录。
- **Agent 集成**：`openchatbi/agent_graph.py` 注册新 Tool；`openchatbi/prompts/agent_prompt.md` 补充下钻分析使用指引。
- **依赖**：仅使用现有 `numpy`、`pandas`，无新增外部依赖。
- **参考代码**：`openchatbi/prompts/analysis/adtributor_demo.py` 为算法来源，实现完成后可 deprecate。
- **非目标（本变更不包含）**：自动 SQL 生成以拉取各维度 baseline/actual 数据（由 Agent 通过 Text2SQL 预先查询后传入）；时序异常检测算法本身（README 中另一项能力，独立实现）。
