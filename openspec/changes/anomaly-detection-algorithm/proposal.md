## Why

目前系统使用时间序列预测，但缺乏强大的异常评估机制。通过引入异常评估模块，基于多个因素（如重构误差、历史上下文、持续时间）对异常进行 0 到 1 的打分，我们可以显著提高异常检测的准确性，减少误报，并更好地识别对业务影响较大的问题。

## What Changes

- 添加一个异常评估模块，调用 `call_timeseries_service` 获取预测基准并计算异常分数（0 到 1）。
- 实现评估窗口，同时评估最近的几个时刻（越新权重越大），以减少噪音。
- 结合一组**相互正交**的评估因子：偏离显著性（基于稳健噪声尺度的 z 分数，统一了上下界与重构误差）、偏离方向（下降/上涨可分别加权）、业务体量调制（基于期望水平）、历史噪声频率阻尼、异常持续时间。
- 将异常评估集成到一个新的工具中（`tool` 层薄封装 + `analysis` 层核心算法），供 Agent 调用。

## Capabilities

### New Capabilities
- `anomaly-detection`: 一项新工具/能力，利用现有的预测服务并应用多因素打分策略，对时间序列数据进行异常检测。

### Modified Capabilities

## Impact

- **新工具**: 核心算法实现在 `openchatbi/analysis/anomaly_detection.py`，工具封装在 `openchatbi/tool/anomaly_detection.py`。
- **依赖**: 依赖现有的 `timeseries_forecast` 工具及其底层服务。
- **Agent 能力**: Agent 将不仅能够预测，还能明确检测并对时间序列数据中的异常进行打分。
