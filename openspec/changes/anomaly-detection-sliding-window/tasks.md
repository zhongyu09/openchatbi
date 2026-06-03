## 1. 下层打分语义修正（evaluation_window）

- [x] 1.1 将 `_evaluate_window` 的窗口分数语义改为**归属窗口最后一个点（anchor）**，前点作为上下文做轻度平滑 + 持续性佐证
- [x] 1.2 保留线性"越新权重最大"作为轻度平滑核，并将平滑强度（anchor 权重占比）暴露为可调参数
- [x] 1.3 **重要**：确保实现中所有代码注释保留为英文（Keep comments in English）

## 2. 分块预测（block forecasting）

- [x] 2.1 实现按 `stride` 的非重叠分块预测：每块用"它之前"的历史，horizon = `stride`，拼接成整段 predicted
- [x] 2.2 实现 `stride` 默认值的 frequency 映射（minute-level=12(TBD) / hourly=12 / daily=7 / weekly=monthly=4 / 其他=10），`stride` 可选并允许显式覆盖
- [x] 2.3 优雅处理单次分块预测失败/超时（降级或跳过并在报告标注），避免单块失败导致整体崩溃

## 3. 上层滑动扫描与区间汇总（detection_range）

- [x] 3.1 新增上层编排函数（如 `detect_anomaly_range`）：在 `detection_range` 上滑动下层窗口，生成逐点 severity 曲线
- [x] 3.2 实现滞回阈值切分（`t_high` 开启 / `t_low` 结束）与相邻区间 `gap` 合并
- [x] 3.3 实现区间接受准则：area ≥ `A_min` 或 peak ≥ `P_high`（取代单纯 min-duration）
- [x] 3.4 每个区间输出起止时间/索引、peak、mean、area、方向（drop/rise）、贡献点明细

## 4. 工具集成与兼容性

- [x] 4.1 `AnomalyDetectionInput` 增加 `detection_range`（末尾点数，无默认）、`stride`（可选）参数及说明；**移除对外的 `evaluation_window` 参数**
- [x] 4.2 实现 `evaluation_window` 内部默认值的 frequency 映射（minute-level=6 / hourly=daily=weekly=monthly=3），不对外暴露
- [x] 4.3 不传 `detection_range`（或 `<= evaluation_window`）时退化为现有"末尾单窗口"行为
- [x] 4.4 扩展 `format_anomaly_report`（或新增多区间格式化函数），输出"区间列表 + 总览"
- [x] 4.5 更新 Agent 使用指引/prompt，说明何时用 `detection_range` 做范围扫描

## 5. 测试与阈值探索

- [x] 5.1 为分块预测、anchor 归属打分、滞回切分、区间接受准则编写单元测试
- [x] 5.2 用正常/异常样本（含单点尖异常、持续异常、多段异常）验证召回与误报
- [ ] 5.3 **实测探索**区间切分阈值（`t_high`/`t_low`/`A_min`/`P_high`/`G`）与下层平滑强度的最佳默认值（需接入真实预测服务 + 标注样本，留作后续调参）
- [x] 5.4 验证向后兼容：不传 `detection_range` 时输出与现有实现一致
