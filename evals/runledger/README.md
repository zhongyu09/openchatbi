# RunLedger eval (OpenChatBI)

This suite is **replay-only** by default. It runs a deterministic CI check using a JSONL adapter that proxies tool calls through RunLedger and replays results from a cassette.

The `eval` extra pins RunLedger to **0.2.4**, and CI installs that same extra. The adapter builds and invokes OpenChatBI's real main agent graph, while model decisions are scripted and tool operations return cassette results. SQL generation, live retrieval, MCP, memory, and the data-analysis sub-agent are replaced or disabled. These checks cover graph routing and protocol contracts, not SQL or answer quality.

## Run (replay)

```bash
python -m pip install -e ".[eval]"
runledger run evals/runledger --mode replay --baseline baselines/runledger-openchatbi.json
```

The schema-discovery case also requires a `search_schema` call, so a valid final JSON response alone cannot satisfy that case.

## Verify a deliberate regression

```bash
python -m evals.runledger.check_regression
```

This runs `c21_schema_discovery` twice using temporary suites and the current Python interpreter. The unchanged adapter must pass with exactly one `search_schema` call. A temporary wrapper then changes only that prompt's scripted trajectory to skip the call and invokes the same graph entrypoint. The final output must still match the JSON schema, while RunLedger must exit 1 with a `must_call` failure naming `search_schema`.

The checker itself exits 0 only when both outcomes are verified. It retains each run's standard artifacts and `evidence.json` under `runledger_out/schema-regression-check/`. A startup error, timeout, or unrelated failure does not count as detecting the regression. No committed adapter, case, cassette, or baseline is modified by the check.

## Record / update cassette (optional)

Recording is separate from this replay-only upgrade check. Most handlers in `tools.py` invoke real tools and require a fully configured OpenChatBI environment (valid `openchatbi/config.yaml`, data warehouse/catalog, LLM keys). The `text2sql` handler is an echo stub with empty SQL/data, so recording this suite does not capture real SQL generation. Review and adapt that handler before using newly recorded SQL results as evidence.

```bash
runledger run evals/runledger --mode record \
  --baseline baselines/runledger-openchatbi.json
```

Notes:
- `suite.yaml` already sets `tool_module: evals.runledger.tools`.
- Tool args are passed as JSON objects; see `evals/runledger/cassettes/t1.jsonl` for the exact shape.
- After recording, promote the new baseline:

```bash
runledger baseline promote \
  --from runledger_out/runledger-openchatbi/<run_id> \
  --to baselines/runledger-openchatbi.json
```

## Case / cassette / protocol format

### Case YAML (`cases/*.yaml`)
- `id`, `description`, `input.prompt`, `cassette` — consumed by RunLedger.
- RunLedger rejects unknown fields, so these files intentionally do
  not contain judge-only metadata such as `category` or `gold`.
- Judge cases live separately under `evals/judge/example_cases/`.

### Cassette JSONL (`cassettes/*.jsonl`)
One line per tool call, in trajectory order. Shape:
`{"tool": <name>, "args": {...}, "ok": true, "result": <any>}`.
The `args` must match what the agent emits after JSON canonicalization: object key order and whitespace do not matter. RunLedger 0.2.4 consumes each matching tool-name/argument entry once per case, so repeated identical calls need one entry per call.

### Protocol note: keying on prompt
The JSONL `task_start` message carries a `task_id`, which the adapter uses for the graph thread. The scripted model driver (`agent/agent.py:_scripted_llm_call`)
keys the scripted trajectory on the **user prompt text** (`_TRAJECTORIES`) and advances by the
count of `ToolMessage`s already in history. Adding a RunLedger case therefore requires
(1) a `cases/*.yaml`, (2) a matching `cassettes/*.jsonl`, and (3) a `_TRAJECTORIES[prompt]` entry.

### Adding a new case
1. Add a `_TRAJECTORIES["your prompt"]` entry in `evals/runledger/agent/agent.py`.
2. Create `evals/runledger/cases/<id>.yaml` with only RunLedger-supported fields.
3. Create `evals/runledger/cassettes/<id>.jsonl` with one JSONL line per tool call in trajectory order. The `args` fields must match what `_TOOL_ARGS_BUILDERS` emit for that prompt.
4. If the case should also be judged by LLM-as-Judge, add a separate `evals/judge/example_cases/<id>.yaml`.
5. Run `uv run pytest tests/eval/test_runledger_agent.py -v` to verify the new case passes the consistency checks.

### Running unit tests (no runledger required)
```bash
uv run pytest tests/eval/test_runledger_agent.py -v
```
These tests verify the scripted driver, trajectory uniqueness, corpus size, and gold/driver consistency — all without the `runledger` package installed.

### Full replay (CI only, requires eval extra)
Install the eval extra: `pip install -e ".[eval]"`, then:
```bash
runledger run evals/runledger --mode replay --baseline baselines/runledger-openchatbi.json
```
This runs in CI for PRs to `main` (see `.github/workflows/`) or by manual dispatch.

