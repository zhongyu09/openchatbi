## Context

现有异常检测（见 `anomaly-detection-algorithm` change）只评估序列末尾的一个短窗口，无法分析一段较长时间范围、也无法定位"哪几段是异常"。直接把长范围当成一个大窗口评估会被双重平滑（远端多步预测衰减 + 窗口内贡献被稀释）而漏报。本变更引入**双层评估窗口**：上层 `detection_range` 在长范围上滑动复用下层 `evaluation_window` 的短窗口检测，并把逐点结果汇总为一个或多个异常区间。

## Goals / Non-Goals

**Goals:**
- 在长范围（`detection_range`）上做滑动异常检测，输出一个或多个异常区间。
- 通过**分块预测**（horizon = `stride`）兼顾三点：远端预测保真、调用次数可控（`ceil(N/stride)`）、块内异常不进模型输入以防污染漏报。
- 修正下层打分语义：分数归属窗口最后一个点，前点做降噪/佐证。
- 上层用滞回切分 + 面积/峰值准则汇总区间，比单纯 min-duration 更平衡精确率与召回。
- 向后兼容现有"末尾单窗口"调用。
- **代码规范**：实现中所有代码注释保留为英文（Keep comments in English）。

**Non-Goals:**
- 替换现有 `timeseries_forecast` 工具。
- 引入监督式 ML 打分。
- 对持续时间远超 `stride` 的超长异常做全程打满分（仅保证 onset / 起始区间被检出，见 Risks）。

## Decisions

### 1. 双层窗口与命名

- **上层 `detection_range`（检测范围）**：要分析的长跨度（点数）。语义为**末尾 `detection_range` 个点**：`history = input_data[:-detection_range]`，`scan = input_data[-detection_range:]`，其前的点为预测/上下文历史。本身不参与单点计算，是滑动扫描的范围 + 区间汇总层。不给默认值，由用户指定；不传（或 `<= evaluation_window`）则退化为单窗口。
- **下层 `evaluation_window`（评估窗口）**：短小的局部打分单元。**不对外暴露为工具参数**（避免调用方 LLM 与 `detection_range` 混淆），改为纯内部值，仅按 `frequency` 取默认（见下表）。`analysis` 层函数仍保留该参数以便测试/程序化调用。
- 二者是**独立旋钮**：`evaluation_window` 管局部打分平滑，`stride` 管预测保真，`detection_range` 管扫描范围。

#### frequency 驱动的默认值

`frequency` 现仅透传给预测服务；本变更让它额外驱动 `evaluation_window` 与 `stride` 的默认值（粒度越细噪声越多 → `evaluation_window` 越长；数据点越少/季节越长 → `stride` 越小）：

| frequency | `evaluation_window`（下层 w，内部） | `stride`（= forecast horizon） |
|-----------|------------------------------|-------------------------------|
| minute-level (1/5/10min) | 6 | 12（TBD，依模型实测） |
| hourly    | 3 | 12 |
| daily     | 3 | 7  |
| weekly    | 3 | 4  |
| monthly   | 3 | 4  |
| 其他 fallback | 3 | 10 |

### 2. 分块预测（block forecasting，horizon = `stride`，非重叠）

```
预测阶段（约 ceil(N/stride) 次服务调用）：
  block_k: 用 [.. a_k) 历史 ── forecast stride 步 ──▶ 预测 [a_k .. a_k+stride)
           a_{k+1} = a_k + stride，每块都只用"它之前"的历史
  → 拼接成整段 predicted[0..N)

打分阶段（纯本地，不再调服务）：
  evaluation_window(=3) 在 actual/predicted 上逐点滑动 → 逐点 severity 曲线
```

- **为什么是 `stride` 块而非"一次长预测"或"逐点重预测"**：
  - 一次性预测 N 步：远端预测回归趋势/均值，异常基准被平滑 → 漏报。
  - 逐点重预测：成本 O(N)；且把上一刻异常实际值喂进历史，模型"学会"异常 → 偏差消失 → 漏报（污染）。
  - 分块（horizon = `stride`）：调用降到 `ceil(N/stride)`；horizon 短，远端衰减受限；块内点用块之前的干净历史预测，块内异常不进入输入。
- **`stride` 默认值**：按 `frequency` 取值，见第 1 节的「frequency 驱动的默认值」表（hourly=12 / daily=7 / weekly=monthly=4 / minute-level=12(TBD) / 其他=10）。`stride` 可选，可被显式覆盖。
- **选 `stride` 的原则**：`stride` 应 ≥ 你关心的最短异常时长（让一段异常尽量落在"用干净历史预测出来的同一块"里），又要足够小以免远端平滑。

### 3. 下层打分语义：分数归属窗口最后一个点（anchor）

- 窗口分数归属于窗口的**最后一个点（anchor）**；前面 `w-1` 个点作为上下文，用于轻度平滑/降噪与持续性佐证。
- 逐点滑动（stride=1 的打分滑动，区别于第 2 节的预测分块）后，**每个点恰好作为 anchor 被打分一次** → 每点一个 severity 值 → **不存在重叠窗口的聚合歧义**（原先"窗口分摊到多点 + 取 max/mean"的问题随之消失）。
- 保留现有**线性"越新权重越大"**作为**轻度**平滑核：anchor 仍占主导，前点以递减权重把 anchor 分数平滑一下，避免把孤立单点尖峰直接抹平；保留 duration 加成作为持续性佐证。
- **去噪分工（方案 C 折中）**：下层只做轻度平滑；"过滤孤立噪声尖峰"的主力交给上层区间汇总，避免下层重平滑与上层过滤双重抑制导致漏报。
- **可调旋钮**：平滑强度（anchor 权重占比）暴露为可调参数，便于按指标在"降噪 vs 灵敏"之间调节。

### 4. 上层区间汇总：滞回切分 + 面积/峰值准则

对逐点 severity 曲线：

1. **滞回阈值切分（hysteresis / 施密特触发）**：severity 上穿 `t_high` 才开启区间，回落到 `t_low`（`t_low < t_high`）以下才结束。抗抖动、少碎片。
2. **gap 合并**：相邻区间间隔 ≤ `G` 个点则合并（容忍异常中途的瞬时回正）。
3. **接受准则（取代单纯 min-duration）**：保留某区间当且仅当
   - **area（区间内 severity 之和）≥ `A_min`**，**或**
   - **peak（区间内最大 severity）≥ `P_high`**（高峰值豁免：极尖锐的短异常也保留）。

   面积把"幅度 × 持续"合成一个量：短而剧烈、长而温和都能过，短而温和的噪声被挡。`min-duration` / `gap` 退为次要结构参数。
4. **每个区间输出**：起止时间/索引、peak 分、mean 分、area、方向（drop/rise，由区间内主导方向决定）、贡献点明细。

### 5. 向后兼容与输出

- 不传 `detection_range`（或 `detection_range <= evaluation_window`）时，退化为现有"末尾单窗口"行为与输出。
- 传 `detection_range` 时，输出扩展为"异常区间列表（可能多个）+ 总览"。`format_anomaly_report` 扩展或新增多区间格式化函数。

## Open Questions / TBD（需实测探索）

- **区间切分阈值**：`t_high`、`t_low`、`A_min`、`P_high`、`G` 的最佳默认值需用正常/异常样本实测确定（先给保守初值，迭代调优）。`ANOMALY_THRESHOLD=0.6` 可作为 `t_high` 的初始参考。
- **去噪强度取向**：已定方案 C（下层轻平滑 + 上层 area/滞回过滤）。若实测发现漏报偏高/误报偏高，再在 C 的旋钮上调整，或局部退向 A（下层重降噪）/ B（下层几乎不平滑）。
- **frequency 驱动的默认值**：上表（`evaluation_window` 与 `stride`）为初值，按实测调整；minute-level 的 `stride` 尤其需依模型实测。

## Risks / Trade-offs

- **风险：异常持续时间 > `stride` 时，后续块的锚点历史包含前一块的异常 → 尾部被"学进去"漏报。**
  - **取舍（已接受）**：方法对 onset / 区间起始最敏感；持续异常本会有终止点，关注起始即可，不追求对无限持续异常全程打满分。
- **风险：滞回/面积阈值需按指标调参。**
  - **缓解**：阈值与下层平滑强度均暴露为可调参数，给出合理初值并支持实测迭代。
- **风险：分块边界处的预测精度在每块远端较低。**
  - **缓解**：`stride` 取适中值；逐点 severity + 区间汇总对单点抖动有容忍。
- **风险：底层预测服务失败/超时（多次分块调用放大概率）。**
  - **缓解**：优雅处理单次调用失败，返回信息丰富的错误；必要时对失败块降级或跳过并在报告中标注。
