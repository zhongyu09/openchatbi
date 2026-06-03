## Why

现有的 `anomaly-detection` 能力（`openchatbi/analysis/anomaly_detection.py`）只评估时间序列**末尾的一个短窗口**：用历史 forecast 出最后 `evaluation_window`（默认 3）个点，逐点打分后聚合成单一分数。这在实践中暴露两个问题：

1. **缺少对一段较长范围的扫描能力**：用户常希望分析一整段时间（例如 24 小时 hourly 粒度），找出其中"哪几段"是异常，而不是只看末尾一刻。
2. **长窗口会被平滑**：如果直接把长范围当成一个大窗口评估，会同时遭遇两重平滑——(a) 一次性多步预测时远端预测回归趋势/均值，异常点的期望基准本身就糊了；(b) 逐点贡献被窗口内大量正常点稀释。结果是真实异常被抹平、漏报。

因此需要补上"上层"：在长范围上**滑动**地复用短窗口检测，并把逐点结果**汇总成一个或多个异常区间**。

## What Changes

- 引入**双层评估窗口**概念并显式命名：
  - **上层 `detection_range`（检测范围）**：要分析的长跨度（如 24h）。语义为**末尾 `detection_range` 个点**（`history = input_data[:-detection_range]`，`scan = input_data[-detection_range:]`），它之前的点是预测/上下文历史。它本身不直接参与单点计算，而是滑动扫描的范围，并负责把逐点结果汇总成异常区间。`detection_range` 不给默认值，由用户按要分析的范围指定；不传则退化为单窗口。
  - **下层 `evaluation_window`（评估窗口）**：短小的局部 forecast 对比 + 逐点打分计算单元。**不再对外暴露为工具参数**（避免调用方 LLM 与 `detection_range` 混淆），改为纯内部值，仅按 `frequency` 取默认（粒度越细噪声越多、窗口越长，见 design）。
- **分块预测策略（block forecasting，horizon = `stride`，非重叠）**：在 `detection_range` 上按 `stride` 分块预测，约 `ceil(N/stride)` 次调用。每块用"它之前"的历史预测，块内异常不进入模型输入，避免**滚动重预测把异常学进去导致的漏报**；同时用较短 horizon 避免一次性长预测的远端平滑。`stride` 同样**可选**，不传时按 `frequency` 取默认（见 design）。
- **`frequency` 的新职责**：现有 `frequency` 仅被透传给预测服务；本变更让它**额外驱动 `evaluation_window` 与 `stride` 的默认值**。
- **下层打分语义修正：窗口分数归属到窗口的最后一个点（anchor）**，前面的点作为上下文做轻度平滑/降噪 + 持续性佐证。逐点滑动后，每个点恰好被打分一次，天然形成一条逐点 severity 曲线（不再有重叠窗口的聚合歧义）。
- **上层区间汇总**：将逐点 severity 曲线用**滞回阈值（hysteresis）切分**为连续异常区间，相邻近区间按 `gap` 合并，并用**面积（累积严重度）+ 高峰值豁免**作为区间接受准则（取代单纯的 min-duration 长度过滤），输出**一个或多个**异常区间。
- 工具入参新增 `detection_range`、`stride`；不传 `detection_range` 时退化为现有"末尾单窗口"行为，保持向后兼容。
- 输出从"单一 score + 一份报告"扩展为"异常区间列表（可能多个）+ 总览"。

## Capabilities

### New Capabilities

（无全新能力，本变更在现有 `anomaly-detection` 能力上扩展滑动检测与区间汇总。）

### Modified Capabilities

- `anomaly-detection`: 在原有单窗口打分基础上，新增上层 `detection_range` 滑动扫描、分块预测、逐点 severity 曲线与多区间汇总。原单窗口行为作为向后兼容的退化路径保留。

## Impact

- **核心算法**：`openchatbi/analysis/anomaly_detection.py` 新增上层编排函数（如 `detect_anomaly_range`）与区间切分逻辑；下层 `_evaluate_window` 调整为"分数归属末点"语义。
- **工具封装**：`openchatbi/tool/anomaly_detection.py` 的 `AnomalyDetectionInput` 增加 `detection_range`、`stride` 参数，并**移除对外的 `evaluation_window` 参数**（改为内部按 `frequency` 推导）；输出报告格式扩展为多区间。
- **依赖**：继续依赖现有 `timeseries_forecast` 服务，不新增外部依赖。
- **Agent 能力**：Agent 可对一整段时间范围做异常扫描，并定位到具体的异常区间（起止时间、严重度、方向）。
- **代码规范**：实现中所有代码注释保留为英文（Keep comments in English）。
- **非目标**：不替换现有 `timeseries_forecast`；不在本期引入监督式 ML 打分；对"持续时间长于 `stride` 的超长异常"，仅保证检测其 onset / 起始区间，不追求对无限持续异常全程打满分。

