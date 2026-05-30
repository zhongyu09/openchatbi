## 1. Core Algorithm Setup

- [x] 1.1 Create `openchatbi/analysis/adtributor.py` and `openchatbi/analysis/models.py`
- [x] 1.2 Define Pydantic models for Input/Output (DimensionResult, AdtributorOutput) in `models.py`
- [x] 1.3 Migrate Surprise and EP math functions (`add_surpise`, `add_explanatory_power`) from demo to `adtributor.py`, removing external dependencies

## 2. Core Algorithm Implementation

- [x] 2.1 Implement core `adtributor` function handling `dict[str, DataFrame]` input for absolute metrics (`derived=False`)
- [x] 2.2 Implement Taylor series derived metrics logic in `adtributor` function (`derived=True`)
- [x] 2.3 Implement sorting and threshold filtering logic (tep, teep, additional_check)

## 3. Tool Interface Implementation

- [x] 3.1 Create `openchatbi/tool/adtributor_tool.py`
- [x] 3.2 Define Tool input schema (Melted Table format) and output schema (Narrative-enhanced)
- [x] 3.3 Implement data transformation logic: 1D long list of dicts -> `dict[str, DataFrame]`
- [x] 3.4 Implement Narrative generation logic (translate EP/Surprise into readable text)

## 4. Agent Integration & Testing

- [x] 4.1 Register `adtributor_drilldown` tool in `openchatbi/tool/mcp_tools.py` / `agent_graph.py` (or appropriate tool registry)
- [x] 4.2 Update `openchatbi/prompts/agent_prompt.md` to guide the Agent on when and how to use the drilldown tool (including Text2SQL preparation)
- [x] 4.3 Write unit tests in `tests/analysis/test_adtributor.py` covering absolute and derived metric scenarios
- [x] 4.4 Mark `openchatbi/prompts/analysis/adtributor_demo.py` as deprecated (add warning comment at top)