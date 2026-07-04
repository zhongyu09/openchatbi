# Data Analysis Artifact Store Enhancement Design

## Problem

当前数据分析工具链依赖 agent 在工具调用之间传递完整数据，或者从自然语言工具输出中抽取结构化结果。例如：

- `text2sql` 返回查询结果后，agent 需要把数据重新组织成下游工具输入。
- `timeseries_forecast` 返回人类可读文本，agent 需要从文本中解析预测值。
- `adtributor_drilldown` 需要完整 melted rows，容易出现格式错误、数值丢失或上下文截断。
- 如果需要预测值和实际值 join、按异常窗口聚合、按维度 melt，agent 可能会调用 `run_python_code` 临场处理，稳定性不足。

这类数据传递方式把关键数据正确性依赖在 prompt 和 LLM 解析能力上，工程风险较高。

## Proposal

引入一个 DataFrame Artifact Store，统一存储分析过程中的中间数据和结果数据。工具之间不再传递完整 DataFrame，而是传递 `dataset_id` / `artifact_id` 引用。

推荐调用链：

```text
text2sql / 用户输入 / forecast / transform
        -> 写入 Artifact Store
        -> 返回 dataset_id + schema + preview + quality summary
        -> 下游工具通过 dataset_id 读取完整数据
```

例如，`adtributor_drilldown` 的输入可以从完整 rows：

```json
{
  "data": [
    {
      "dimension_name": "device",
      "element_value": "ios",
      "predict": 1000,
      "real": 600
    }
  ],
  "derived": false,
  "issue_type": "drop"
}
```

简化为：

```json
{
  "dataset_id": "df_adtributor_abc123",
  "derived": false,
  "issue_type": "drop"
}
```

## Expected Benefits

1. 减少 LLM 解析和复制数据的风险。
   - 下游工具读取的是 store 中的结构化数据，而不是 agent 从文本中抽取出来的数据。
   - 可以避免预测结果、SQL 结果、聚合结果在上下文传递中被截断或格式污染。

2. 提升工具链可校验性。
   - 每个 artifact 可以记录 `columns`、`dtypes`、`row_count`、`source`、`created_by`、`time_window` 等 metadata。
   - 下游工具可以在读取 `dataset_id` 后做强校验，例如检查 `adtributor_drilldown` 必需字段是否存在。

3. 降低对 `run_python_code` 临场聚合的依赖。
   - join、aggregate、melt、rename、type cast、gap fill 等操作可以由确定性工具完成。
   - Agent 负责选择步骤和参数，工具负责数据处理和校验。

4. 支持更大的中间数据。
   - 全量数据不需要全部进入 LLM 上下文。
   - LLM 只接收 `dataset_id`、schema、preview 和质量摘要。

5. 更容易调试和复现。
   - 每个 artifact 都可以保存来源、生成步骤、行列数、预览和校验结果。
   - 出错时可以定位是哪一步数据形态不符合预期。

## Core Components

### 1. Artifact Store

负责保存和读取 DataFrame artifact。

建议能力：

- `put_dataframe(df, metadata) -> dataset_id`
- `get_dataframe(dataset_id) -> DataFrame`
- `describe_dataset(dataset_id) -> schema + preview + row_count + metadata`
- `delete_dataset(dataset_id)`
- `list_datasets(thread_id / run_id)`

存储建议：

- 小数据可以放内存。
- 大数据可以落盘为 Parquet / Arrow / SQLite table。
- Metadata 可以放 SQLite 或内存字典。
- 默认按 thread / run 隔离，不跨用户长期共享。

### 2. Dataset Metadata

每个 dataset 至少应包含：

```json
{
  "dataset_id": "df_abc123",
  "source": "text2sql",
  "created_by": "text2sql",
  "row_count": 120,
  "columns": ["time", "device", "value"],
  "dtypes": {
    "time": "datetime64",
    "device": "string",
    "value": "float64"
  },
  "time_window": {
    "start": "2026-07-03 00:00:00",
    "end": "2026-07-04 00:00:00"
  },
  "preview": [
    {"time": "2026-07-03 00:00:00", "device": "ios", "value": 123}
  ],
  "quality_summary": {
    "missing_values": {},
    "duplicate_rows": 0
  }
}
```

## Proposed Tools

### 1. `create_dataframe`

把非 `text2sql` 来源的数据转成 DataFrame artifact。

适用场景：

- 用户直接粘贴 Markdown 表格、CSV、JSON。
- 用户上传文件。
- 用户手工提供小规模样例数据。
- 外部工具返回结构化数据但还没有写入 artifact store。

输入示例：

```json
{
  "source_type": "inline_table",
  "content": "| date | device | value |\n| --- | --- | --- |\n| 2026-07-03 | ios | 123 |",
  "schema_hint": {
    "date": "datetime",
    "device": "string",
    "value": "float"
  }
}
```

输出示例：

```json
{
  "dataset_id": "df_user_001",
  "row_count": 1,
  "columns": ["date", "device", "value"],
  "preview": [
    {"date": "2026-07-03", "device": "ios", "value": 123}
  ]
}
```

### 2. `describe_dataset`

返回 dataset 的 schema、行数、预览和质量摘要，供 agent 判断下一步。

输入：

```json
{
  "dataset_id": "df_user_001"
}
```

### 3. `transform_dataframe`

提供确定性 DataFrame 变换。

建议支持：

- filter
- group by + aggregate
- join
- melt / pivot
- rename
- select columns
- type cast
- fill missing periods
- sort
- deduplicate

输出新的 `dataset_id`，不覆盖原始数据。

### 4. `forecast_dataset`

输入历史时间序列 dataset，输出预测结果 dataset。

建议支持：

- 单条 series forecast。
- 按 group columns 批量 forecast。
- 返回完整预测点。
- 可选返回聚合值，例如预测窗口 `sum` / `avg`。

输入示例：

```json
{
  "dataset_id": "df_history_001",
  "time_column": "time",
  "target_column": "value",
  "group_columns": ["device"],
  "forecast_window": 24,
  "frequency": "hourly",
  "aggregate": "sum"
}
```

输出可以包含：

```json
{
  "dataset_id": "df_forecast_001",
  "row_count": 48,
  "columns": ["time", "device", "prediction"],
  "aggregate_dataset_id": "df_forecast_agg_001"
}
```

### 5. `prepare_adtributor_drilldown_data`

面向 anomaly drill-down 的高层工具，推荐作为核心增强点。

职责：

- 获取或接收历史数据 dataset。
- 获取或接收异常窗口实际值 dataset。
- 按维度元素生成预测 baseline。
- 将预测窗口聚合成 `predict`。
- 将异常窗口实际值聚合成 `real`。
- 输出 `adtributor_drilldown` 可直接消费的 melted dataset。

输入示例：

```json
{
  "history_dataset_id": "df_history_001",
  "actual_dataset_id": "df_actual_001",
  "time_column": "time",
  "metric_column": "value",
  "dimension_columns": ["device", "province"],
  "anomaly_window": {
    "start": "2026-07-04 10:00:00",
    "end": "2026-07-04 12:00:00"
  },
  "frequency": "hourly",
  "metric_type": "absolute"
}
```

输出：

```json
{
  "dataset_id": "df_adtributor_001",
  "columns": ["dimension_name", "element_value", "predict", "real"],
  "row_count": 20
}
```

### 6. `adtributor_drilldown`

建议新增 `dataset_id` 输入方式。

短期可以兼容当前 `data` 参数：

```json
{
  "dataset_id": "df_adtributor_001",
  "derived": false,
  "issue_type": "drop"
}
```

工具内部读取 dataset 后，校验字段：

- `derived=false`: `dimension_name`, `element_value`, `predict`, `real`
- `derived=true`: `dimension_name`, `element_value`, `predict_numerator`, `predict_denominator`, `real_numerator`, `real_denominator`

## Handling User-Provided Data

如果数据不是来自 `text2sql`，例如用户直接粘贴表格、上传文件或输入 JSON，建议统一通过 `create_dataframe` 转成 artifact。

流程：

```text
用户输入数据
  -> create_dataframe
  -> dataset_id
  -> describe_dataset
  -> transform / forecast / drilldown
```

这样用户输入和数据库查询结果可以进入同一个数据处理链路，下游工具无需关心数据来源。

## Anomaly Drill-Down Flow

推荐目标流程：

```text
1. search_schema / show_schema
2. text2sql 获取历史窗口数据 -> history_dataset_id
3. text2sql 获取异常窗口实际数据 -> actual_dataset_id
4. prepare_adtributor_drilldown_data
   - 按维度元素调用预测服务
   - 聚合预测窗口为 predict
   - 聚合异常窗口实际值为 real
   - 生成 melted rows dataset
5. adtributor_drilldown(dataset_id)
6. 总结根因
```

如果数据由用户提供：

```text
1. create_dataframe -> source_dataset_id
2. transform_dataframe 拆分历史窗口和异常窗口
3. prepare_adtributor_drilldown_data
4. adtributor_drilldown(dataset_id)
```

## Open Questions

1. Artifact Store 的生命周期应该是 thread 级、run 级，还是 session 级？
2. 大数据落盘格式优先选 Parquet、SQLite table，还是 Arrow IPC？
3. `text2sql` 是否直接返回 `dataset_id`，还是新增一个 wrapper 工具负责执行 SQL 并注册结果？
4. `run_python_code` 是否允许读取 / 写入 Artifact Store？如果允许，需要限制其权限和可见 dataset。
5. 是否需要 artifact lineage，用于追踪每个 dataset 的父 dataset 和变换步骤？
6. 对派生指标，`prepare_adtributor_drilldown_data` 是否负责 numerator/denominator 的预测和聚合，还是先只支持绝对指标？

## Suggested Implementation Phases

1. Phase 1: 引入最小 Artifact Store。
   - 支持内存存储、metadata、`put/get/describe`。
   - 新增 `create_dataframe` 和 `describe_dataset`。

2. Phase 2: 让 `text2sql` 或其 wrapper 可以注册查询结果。
   - 返回 `dataset_id`、schema、row count 和 preview。
   - 保留当前文本输出兼容现有链路。

3. Phase 3: 改造 `adtributor_drilldown` 支持 `dataset_id`。
   - 保留当前 `data` 参数作为兼容路径。
   - 增加字段校验和错误提示。

4. Phase 4: 新增 `prepare_adtributor_drilldown_data`。
   - 先支持绝对指标。
   - 固化预测、对齐、聚合和 melted rows 生成逻辑。

5. Phase 5: 支持派生指标和 batch forecast。
   - 增加 numerator/denominator 处理。
   - 增强预测服务或预测工具支持按 group 批量预测和聚合输出。
