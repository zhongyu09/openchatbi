# Data Analysis Package (`openchatbi.analysis`)

> 中文版见 [README_cn.md](./README_cn.md).

This package provides OpenChatBI's advanced data analysis capabilities: a
specialized **Data Analysis Agent** that orchestrates analysis tools into
multi-step workflows, plus the two analysis **algorithms** it relies on —
time series **anomaly detection** and multi-dimensional **anomaly drill-down**
(Adtributor).

## Layout

```
openchatbi/analysis/
├── agent.py             # Data Analysis Agent + the `data_analysis` delegation tool
├── anomaly_detection.py # Anomaly detection scoring algorithm (core)
├── adtributor.py        # Adtributor root-cause / drill-down algorithm (core)
└── models.py            # Pydantic output models for the Adtributor algorithm

openchatbi/tool/
├── anomaly_detection.py # `anomaly_detection` tool wrapper (LLM-facing)
└── adtributor_tool.py   # `adtributor_drilldown` tool wrapper (LLM-facing)

openchatbi/prompts/
└── data_analysis_prompt.md  # System prompt with the 5 workflow playbooks
```

The convention is: **algorithm core lives in `analysis/`, the LLM-facing tool
wrapper lives in `tool/`, and the agent that orchestrates the tools lives in
`analysis/agent.py`.**

---

## 1. Data Analysis Agent (`agent.py`)

A specialized sub-agent (built on the
[deepagents](https://github.com/langchain-ai/deepagents) framework) that the
main agent delegates complex analysis to via the `data_analysis` tool. Keeping
the orchestration inside a sub-agent isolates the analysis context and prevents
the main agent's context from ballooning with intermediate tool output.

### Public API

- `build_data_analysis_agent(sql_graph, sync_mode=False, llm_provider=None, checkpointer=None, memory_store=None)`
  — assembles the tool set and compiles the agent graph.
- `get_data_analysis_tool(...)` — wraps the agent in a `data_analysis`
  `StructuredTool` (sync or async) that the main agent registers.

### Tool set

| Tool | Source | Notes |
|------|--------|-------|
| `text2sql` | main SQL generation subgraph | reused |
| `timeseries_forecast` | `openchatbi/tool/timeseries_forecast.py` | requires forecast service |
| `anomaly_detection` | `openchatbi/tool/anomaly_detection.py` | requires forecast service |
| `adtributor_drilldown` | `openchatbi/tool/adtributor_tool.py` | no external service |
| `run_python_code` | `openchatbi/tool/run_python_code.py` | no external service |

`timeseries_forecast` and `anomaly_detection` share the forecasting service
health check and are **registered together only when the service is healthy**.

### Supported scenarios (prompt-driven workflows)

| Scenario | Typical flow |
|----------|--------------|
| Single-metric trend forecasting | `text2sql` → `timeseries_forecast` → interpret |
| Single-metric anomaly detection | `text2sql` → `anomaly_detection` → interpret |
| Single-metric anomaly drill-down | `text2sql` (1D melted table) → `adtributor_drilldown` → interpret |
| Multi-metric correlation | `text2sql` → `run_python_code` → interpret |
| Business combination analysis | `text2sql` → `run_python_code` → interpret |

### LLM selection

The agent uses the optional `analysis_llm` configuration when present, otherwise
it falls back to `default_llm` (see `openchatbi.llm.llm.get_analysis_llm`). This
lets you point analysis at a stronger reasoning model without changing the main
agent's model.

### Sub-agent isolation

The agent is a separately compiled graph that may share the main agent's
`checkpointer`. The `data_analysis` tool therefore derives an isolated child
config (`_build_sub_agent_config`):

- a deterministic child `thread_id` (`"{parent_thread_id}:data_analysis"`) so it
  satisfies LangGraph's checkpointer requirement, keeps interrupt/resume stable,
  and never clobbers the main agent's checkpoint thread;
- inherited `checkpoint_ns` / `checkpoint_id` are cleared so the sub-agent starts
  from a clean namespace;
- other config (callbacks, tags, metadata) is propagated unchanged.

`GraphInterrupt` raised inside the sub-agent is re-raised (not swallowed) so
human-in-the-loop interrupts can bubble up to the main graph. Final output is
normalized to a string via `_extract_final_content`, including the case where
the model returns multimodal content blocks.

---

## 2. Anomaly Detection (`anomaly_detection.py`)

Scores how anomalous the most recent points of a single-metric time series are,
by comparing them against a forecasting baseline. The forecast comes from the
time series forecasting service; if it is unavailable, detection fails fast with
an error in the details dict.

### Output

`evaluate_anomalies(...)` returns `(score, details)` where `score` is in `[0, 1]`
(closer to 1 = more significant / higher-impact anomaly). `format_anomaly_report`
turns it into a human-readable report (used by the tool wrapper). Severity bands:
`>0.8` Critical, `>0.6` High, `>0.4` Medium, else Low.

### Scoring strategy

The score combines a set of intentionally **orthogonal** factors so the same
evidence is not double-counted:

- **Deviation significance** — how far the actual value is from the forecast in
  robust noise-scale (sigma) units. Significance ramps to 0.5 at 3σ and to 1.0 at
  6σ. The noise scale is estimated from first differences with a MAD-based
  estimator (robust to historical spikes; not inflated by seasonality).
- **Direction weighting** — drops vs rises can be weighted differently
  (`drop_weight` / `rise_weight`), since an unexpected drop is often more severe.
- **Volume modulation** — anomalies on high-traffic moments matter more. A
  multiplier in `[0.6, 1.0]` driven by the *expected* (predicted) level, so a
  drop-to-zero is never penalized by it.
- **Historical anomaly frequency** — intrinsically noisy/jumpy metrics are
  dampened (fewer false positives).
- **Duration** — a run of consecutive anomalous points near the end of the window
  boosts the score.

Recent points are weighted more heavily (linear weighting). Per-point sigmas from
the forecast service (`prediction_std`) are used when available, otherwise the
historical noise-scale estimate is used.

### Input contract

`input_data` is a list of numbers or dicts; the **last `evaluation_window`
points are the points to evaluate**, the preceding points are the historical
context (used for forecast input and for the noise/volume/frequency factors).
`input_data` length must be greater than `evaluation_window`.

---

## 3. Anomaly Drill-Down / Adtributor (`adtributor.py`, `models.py`)

Multi-dimensional root-cause analysis based on Microsoft's Adtributor algorithm.
Given baseline (predict) vs actual (real) values broken down by dimension
elements, it finds the dimensions and elements that best explain an anomaly.

### Core concepts

- **Surprise** — a Jensen-Shannon-divergence-style measure of how much an
  element's distribution shifted between predict and real.
- **Explanatory Power (EP)** — the share of the overall metric change attributable
  to an element. For derived (ratio) metrics, EP is computed with the ratio
  decomposition and normalized to sum to 1.
- **Thresholds** — `tep` (cumulative EP threshold to accept a dimension's top
  elements as root cause) and `teep` (per-element EP threshold to be considered),
  plus `k` (number of top candidate dimensions to return).
- **`additional_check`** — guards against degenerate explanations (e.g. when the
  candidate elements already account for ~100% of the value).

### Absolute vs derived metrics

- **Absolute** metrics use `predict` / `real`.
- **Derived (ratio)** metrics use `predict_numerator` / `predict_denominator` /
  `real_numerator` / `real_denominator` (set `derived=True`).

### Output (`AdtributorOutput`)

- `root_causes` — `{dimension: [elements]}` for the top-`k` explaining dimensions.
- `ranked_dimensions` — all analyzed dimensions sorted by total surprise.
- `dimension_details` — per-dimension `DimensionResult` (EP, surprise, elements,
  reason).
- `status` — one of:
  - `success` — root cause elements identified;
  - `no_root_cause` — anomaly is systemic / evenly distributed, no element passed
    the thresholds;
  - `no_anomaly_direction` — no element matched the requested `issue_type`
    (`drop` / `rise`).

### Input contract (tool layer)

The `adtributor_drilldown` tool accepts a **1D melted table** (`list[dict]`) with
keys `dimension_name`, `element_value`, and the relevant predict/real fields. The
tool reshapes it into the per-dimension `dict[str, DataFrame]` the core algorithm
expects, runs the algorithm, and attaches a business-friendly narrative to the
result. The prompt instructs the agent to prepare this melted table via
`text2sql` before drilling down.

---

## See also

- API reference: the docs *Tools and Utilities* page documents `agent.py` and the
  two tool wrappers.
- Prompt / workflows: `openchatbi/prompts/data_analysis_prompt.md`.
- Forecasting service: `timeseries_forecasting/README.md`.
