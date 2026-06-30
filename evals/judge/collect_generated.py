from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import yaml


def load_cases(cases_dir: str) -> list[dict[str, Any]]:
    """Read every ``*.yaml`` in ``cases_dir`` and return the parsed cases.

    Each case must have at least an ``id`` and the prompt at
    ``case["input"]["prompt"]``. A ``gold`` key is NOT required, so this works
    on caseless prompt-only files too. Sorted by filename for determinism.
    """
    cases: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.yaml"))):
        with open(path) as fh:
            case = yaml.safe_load(fh) or {}
        cases.append(case)
    return cases


def extract_sql_from_state(state: dict[str, Any], sql_key: str = "sql") -> str:
    """Pull the committed SQL out of an agent state dict.

    PURE — never raises. The state dict is synthesized by
    :func:`_state_from_graph` from the stream updates; if the agent paused for
    human-in-the-loop the dict carries ``__interrupt__`` and no SQL was
    committed, so return an empty string. Otherwise return the (stripped)
    value at ``sql_key``.
    """
    if not isinstance(state, dict):
        return ""
    if "__interrupt__" in state:
        return ""
    return (state.get(sql_key) or "").strip()


def collect(
    cases: list[dict[str, Any]],
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    sql_key: str = "sql",
    on_update: Callable[[list[dict[str, Any]]], None] | None = None,
    on_case_start: Callable[[dict[str, Any], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run ``runner`` over each case and collect the generated SQL.

    ``runner(case) -> state`` is injected so tests need no LLM and no graph.
    Per-case isolation: ANY exception from ``runner`` records an empty SQL for
    that case and processing continues. ``on_update`` is called after each
    record, allowing CLI runs to persist partial progress. ``on_case_start`` is
    called immediately before each case is executed.
    """
    records: list[dict[str, Any]] = []
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        case_id = case.get("id", "")
        prompt = (case.get("input") or {}).get("prompt", "")
        if on_case_start is not None:
            on_case_start(case, idx, total)
        try:
            state = runner(case)
            generated_sql = extract_sql_from_state(state, sql_key=sql_key)
        except Exception:
            generated_sql = ""
        records.append({"id": case_id, "prompt": prompt, "generated_sql": generated_sql})
        if on_update is not None:
            on_update(records)
    return records


# Nodes whose stream update carries the committed SQL (see openchatbi.streaming).
_SQL_NODE_NAMES = ("generate_sql", "regenerate_sql")


def _state_from_graph(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Drive ``graph`` for one case and return a state dict with the final SQL.

    text2sql runs as a tool/subgraph, so the committed SQL never lands in the
    main graph's terminal state (``graph.invoke()`` filters on
    ``output_schema=OutputState``, and ``get_state().values["sql"]`` stays
    empty). The SQL is emitted in the streaming ``updates`` for the
    ``generate_sql`` / ``regenerate_sql`` nodes — exactly what ``run_cli`` and
    ``openchatbi.streaming`` read. We stream with ``subgraphs=True`` and keep the
    last non-empty ``sql`` (so a regenerate-after-retry wins). A pause surfaces
    as an ``__interrupt__`` update, which we forward so
    :func:`extract_sql_from_state` returns ``""``.

    Pure w.r.t. ``graph``: tests inject a fake graph exposing ``stream``.
    """
    case_id = str(case.get("id", ""))
    prompt = (case.get("input") or {}).get("prompt", "")
    run_id = f"eval-{case_id}" if case_id else "eval-case"
    cfg = {
        "configurable": {"thread_id": run_id, "user_id": run_id},
        "metadata": {"user_id": run_id, "session_id": case_id, "request_id": run_id},
        "run_name": f"openchatbi-eval:{run_id}",
    }
    last_sql = ""
    interrupted = False

    from openchatbi.observability.context import current_request_id, current_user_id

    user_token = current_user_id.set(run_id)
    request_token = current_request_id.set(run_id)
    try:
        for _namespace, update in graph.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=cfg,
            stream_mode="updates",
            subgraphs=True,
        ):
            if not isinstance(update, dict):
                continue
            if "__interrupt__" in update:
                interrupted = True
                continue
            for node_name, node_output in update.items():
                if node_name in _SQL_NODE_NAMES and isinstance(node_output, dict):
                    sql = node_output.get("sql")
                    if sql:
                        last_sql = sql
    finally:
        current_request_id.reset(request_token)
        current_user_id.reset(user_token)

    state: dict[str, Any] = {"sql": last_sql}
    if interrupted:
        state["__interrupt__"] = True
    return state


def build_agent_runner(config_path: str, provider: str | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the sync agent graph ONCE and return a ``runner(case)->state`` closure.

    Mirrors ``run_cli.build_sync_graph`` exactly. This is the only part that
    needs a real LLM key; it is kept lazy/separate so importing this module and
    the pure functions above never touches the network.
    """
    # Load .env first so ANTHROPIC_API_KEY / OPENAI_API_KEY (and CONFIG_FILE)
    # are available before openchatbi instantiates the LLM at import time.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Point config at the requested yaml BEFORE importing openchatbi (which
    # calls config.load() at import time reading $CONFIG_FILE), then re-load
    # defensively.
    os.environ["CONFIG_FILE"] = config_path

    from langgraph.checkpoint.memory import MemorySaver

    from openchatbi import config
    from openchatbi.agent_graph import build_agent_graph_sync
    from openchatbi.tool.memory import get_sync_memory_store

    config.load(config_path)

    graph = build_agent_graph_sync(
        config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=get_sync_memory_store(),
        llm_provider=provider,
    )

    return lambda case: _state_from_graph(graph, case)


def write_output(records: list[dict[str, Any]], out_path: str, fmt: str) -> None:
    """Write collected records to ``out_path`` in ``json`` or ``jsonl`` form.

    - ``json``: a ``{id: generated_sql}`` object map.
    - ``jsonl``: one ``{"id","prompt","generated_sql"}`` object per line.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    if fmt == "json":
        obj = {r["id"]: r["generated_sql"] for r in records}
        with open(tmp_path, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp_path, out_path)
    elif fmt == "jsonl":
        with open(tmp_path, "w") as fh:
            for r in records:
                fh.write(
                    json.dumps(
                        {"id": r["id"], "prompt": r["prompt"], "generated_sql": r["generated_sql"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp_path, out_path)
    else:
        raise ValueError(f"unknown format: {fmt!r}")


def write_progress(out_path: str, processed: int, total: int) -> None:
    progress_path = f"{out_path}.progress.json"
    os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
    tmp_path = f"{progress_path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(
            {
                "processed": processed,
                "total": total,
                "complete": processed == total,
                "out": out_path,
            },
            fh,
            indent=2,
        )
    os.replace(tmp_path, progress_path)


def load_existing_output(out_path: str, fmt: str) -> dict[str, str]:
    """Load existing generated SQL keyed by case id for startup reporting."""
    if not os.path.exists(out_path):
        return {}

    with open(out_path) as fh:
        raw = fh.read().strip()

    if not raw:
        return {}

    existing: dict[str, str] = {}
    if fmt == "json":
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return {str(case_id): str(sql or "") for case_id, sql in obj.items()}
        except json.JSONDecodeError as exc:
            print(f"[collect_generated] Warning: Failed to parse existing JSON map: {exc}", file=sys.stderr)
        return {}

    if fmt == "jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[collect_generated] Warning: Skipping malformed JSON line in existing output: {exc}", file=sys.stderr)
                continue
            if isinstance(record, dict) and record.get("id"):
                existing[str(record["id"])] = str(record.get("generated_sql") or "")
        return existing

    raise ValueError(f"unknown format: {fmt!r}")


def records_from_existing(cases: list[dict[str, Any]], existing: dict[str, str]) -> list[dict[str, Any]]:
    """Build ordered output records for cases already present in the output."""
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        if case_id not in existing:
            continue
        prompt = (case.get("input") or {}).get("prompt", "")
        records.append({"id": case_id, "prompt": prompt, "generated_sql": existing[case_id]})
    return records


def merge_records_by_case_order(
    cases: list[dict[str, Any]],
    existing: dict[str, str],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge existing and newly collected records while preserving case order."""
    new_by_id = {str(record["id"]): record for record in new_records}
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        prompt = (case.get("input") or {}).get("prompt", "")
        if case_id in new_by_id:
            records.append(new_by_id[case_id])
        elif case_id in existing:
            records.append({"id": case_id, "prompt": prompt, "generated_sql": existing[case_id]})
    return records


def _short_prompt(prompt: str, max_chars: int = 88) -> str:
    one_line = " ".join(prompt.split())
    if len(one_line) <= max_chars:
        return one_line
    return f"{one_line[: max_chars - 3]}..."


def print_collection_plan(
    cases: list[dict[str, Any]],
    out_path: str,
    fmt: str,
    stream: Any = sys.stderr,
) -> None:
    """Print a friendly startup checklist with existing and pending cases."""
    existing = load_existing_output(out_path, fmt)
    total = len(cases)
    executed = 0
    empty_sql = 0
    pending = 0

    rows: list[tuple[str, str, str]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        prompt = str((case.get("input") or {}).get("prompt", ""))
        if case_id in existing:
            if existing[case_id].strip():
                status = "done"
                executed += 1
            else:
                status = "done, empty SQL"
                empty_sql += 1
        else:
            status = "pending"
            pending += 1
        rows.append((status, case_id, _short_prompt(prompt)))

    print("=== Generated SQL Collection Plan ===", file=stream)
    print(f"Output: {out_path}", file=stream)
    print(
        f"Total: {total} | Done: {executed} | Done with empty SQL: {empty_sql} | Pending: {pending}",
        file=stream,
    )
    if not existing:
        print("Existing output: not found; starting from scratch.", file=stream)
    print("", file=stream)
    for idx, (status, case_id, prompt) in enumerate(rows, start=1):
        print(f"{idx:>3}/{total:<3} [{status}] {case_id}", file=stream)
        if prompt:
            print(f"          {prompt}", file=stream)
    print("========================", file=stream)


def print_case_start(case: dict[str, Any], idx: int, total: int, stream: Any = sys.stderr) -> None:
    """Print a visible progress message before running one case."""
    case_id = str(case.get("id", ""))
    prompt = _short_prompt(str((case.get("input") or {}).get("prompt", "")))
    bar = "=" * 64
    print("", file=stream)
    print(bar, file=stream)
    print(f"RUNNING CASE {idx}/{total}: {case_id}", file=stream)
    if prompt:
        print(f"Prompt: {prompt}", file=stream)
    print(bar, file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.judge.collect_generated")
    parser.add_argument("--cases", default="evals/judge/example_cases")
    parser.add_argument("--config", default="example/config.yaml")
    parser.add_argument("--out", default="generated.json")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json")
    parser.add_argument("--limit", type=int, default=None, help="Only collect the first N cases.")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    print_collection_plan(cases, args.out, args.format, stream=sys.stderr)

    existing = load_existing_output(args.out, args.format)
    total = len(cases)
    existing_records = records_from_existing(cases, existing)
    pending_cases = [case for case in cases if str(case.get("id", "")) not in existing]

    if existing_records:
        print(
            f"Resume: found {len(existing_records)} existing case(s); " f"{len(pending_cases)} case(s) still pending.",
            file=sys.stderr,
        )
    if not pending_cases:
        write_output(existing_records, args.out, args.format)
        write_progress(args.out, processed=len(existing_records), total=total)
        print(
            f"Nothing to run; all {len(existing_records)}/{total} cases are already collected -> {args.out}",
            file=sys.stderr,
        )
        return 0

    original_positions = {str(case.get("id", "")): idx for idx, case in enumerate(cases, start=1)}

    runner = build_agent_runner(args.config)

    def persist(records: list[dict[str, Any]]) -> None:
        merged_records = merge_records_by_case_order(cases, existing, records)
        write_output(merged_records, args.out, args.format)
        write_progress(args.out, processed=len(merged_records), total=total)
        print(f"collected {len(merged_records)}/{total} cases -> {args.out}", file=sys.stderr)

    records = collect(
        pending_cases,
        runner,
        on_update=persist,
        on_case_start=lambda case, _idx, _total: print_case_start(
            case,
            original_positions.get(str(case.get("id", "")), _idx),
            total,
            stream=sys.stderr,
        ),
    )
    if not records:
        write_output(existing_records, args.out, args.format)
        write_progress(args.out, processed=len(existing_records), total=total)

    merged_records = merge_records_by_case_order(cases, existing, records)
    print(f"collected {len(merged_records)} cases -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
