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

