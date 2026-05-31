## 1. 初始设置与模块创建

- [x] 1.1 在 `openchatbi/analysis/` 中创建核心算法，在 `openchatbi/tool/` 中创建工具封装
- [x] 1.2 使用 Pydantic 定义输入 schema `AnomalyDetectionInput`
- [x] 1.3 导入必要的依赖项 (如 `timeseries_forecast`, `requests`, `pydantic`)
- [x] 1.4 **重要**: 确保代码实现中的所有注释保留为英文 (Keep comments in English)

## 2. 核心实现：正交因子异常打分

- [x] 2.1 实现稳健噪声尺度估计（一阶差分 + MAD，标准差回退）与偏离显著性（z 分数 → [0,1]）
- [x] 2.2 实现偏离方向区分与可配置的 `drop_weight`/`rise_weight`
- [x] 2.3 实现基于期望水平的业务体量调制因子（[0.6, 1.0]）
- [x] 2.4 实现历史噪声频率阻尼因子，降低噪声指标的误报
- [x] 2.5 实现评估窗口逻辑（线性权重，越新越大）与连续异常的持续时间加成
- [x] 2.6 实现最终打分函数，将各正交因子组合成 0-1 的分数

## 3. 工具集成

- [x] 3.1 在 `analysis` 层实现 `evaluate_anomalies` 与 `format_anomaly_report`
- [x] 3.2 调用公开方法 `call_timeseries_service` 获取结构化基准预测（预留逐步不确定度接口）
- [x] 3.3 `tool` 层 `anomaly_detection` 仅做薄封装，透传参数并返回格式化报告
- [x] 3.4 格式化输出，提供异常分数和详细报告 (例如，解释分数高/低的原因)

## 4. 测试与优化

- [x] 4.1 为异常打分逻辑编写单元测试
- [x] 4.2 使用样本数据 (正常和异常数据) 测试工具，确保打分行为符合预期
- [x] 4.3 必要时优化打分因素的权重
