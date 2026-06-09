# RunLedger eval (OpenChatBI)

This suite is **replay-only** by default. It runs a deterministic CI check using a JSONL adapter that proxies tool calls through RunLedger and replays results from a cassette.

## Run (replay)

```bash
runledger run evals/runledger --mode replay --baseline baselines/runledger-openchatbi.json
```

## Record / update cassette (optional)

If you want to re-record the cassette with real tool outputs, run in record mode in a fully configured OpenChatBI environment (valid `openchatbi/config.yaml`, data warehouse/catalog, LLM keys).

```bash
runledger run evals/runledger --mode record \
  --baseline baselines/runledger-openchatbi.json \
  --tool-module evals.runledger.tools
```

Notes:
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
- `category` — one of `aggregation | join | timerange | anomaly | visualization | text2sql | report`.
- `gold` — consumed by the out-of-band LLM judge (`evals/judge/`), ignored by RunLedger:
  - `expected_sql` — hand-written gold SQL (empty string for knowledge/schema-only cases).
  - `expected_tool_trajectory` — ordered list of tool names the agent should call.
  - `expected_result_contains` — lowercase substrings expected in the final reply.

### Cassette JSONL (`cassettes/*.jsonl`)
One line per tool call, in trajectory order. Shape:
`{"tool": <name>, "args": {...}, "ok": true, "result": <any>}`.
The `args` must byte-match what the agent emits, because RunLedger replay matches
recorded calls by tool name + args.

### Protocol note: keying on prompt
The JSONL protocol carries no case-id. The agent driver (`agent/agent.py:_scripted_llm_call`)
keys the scripted trajectory on the **user prompt text** (`_TRAJECTORIES`) and advances by the
count of `ToolMessage`s already in history. Adding a case therefore requires (1) a `cases/*.yaml`,
(2) a matching `cassettes/*.jsonl`, and (3) a `_TRAJECTORIES[prompt]` entry.

### Adding a new case
1. Add a `_TRAJECTORIES["your prompt"]` entry in `evals/runledger/agent/agent.py`.
2. Create `evals/runledger/cases/<id>.yaml` with `id`, `category`, `description`, `input.prompt`, `cassette`, and `gold`.
3. Create `evals/runledger/cassettes/<id>.jsonl` with one JSONL line per tool call in trajectory order. The `args` fields must match what `_TOOL_ARGS_BUILDERS` emit for that prompt.
4. Run `uv run pytest tests/eval/test_runledger_agent.py -v` to verify the new case passes the consistency checks.

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
This runs in CI with the `runledger` label gate (see `.github/workflows/`) or nightly.

