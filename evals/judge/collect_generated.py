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
) -> list[dict[str, Any]]:
    """Run ``runner`` over each case and collect the generated SQL.

    ``runner(case) -> state`` is injected so tests need no LLM and no graph.
    Per-case isolation: ANY exception from ``runner`` records an empty SQL for
    that case and processing continues.
    """
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get("id", "")
        prompt = (case.get("input") or {}).get("prompt", "")
        try:
            state = runner(case)
            generated_sql = extract_sql_from_state(state, sql_key=sql_key)
        except Exception:
            generated_sql = ""
        records.append({"id": case_id, "prompt": prompt, "generated_sql": generated_sql})
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
    case_id = case.get("id", "")
    prompt = (case.get("input") or {}).get("prompt", "")
    cfg = {"configurable": {"thread_id": f"eval-{case_id}", "user_id": "eval"}}
    last_sql = ""
    interrupted = False
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
    if fmt == "json":
        obj = {r["id"]: r["generated_sql"] for r in records}
        with open(out_path, "w") as fh:
            json.dump(obj, fh, indent=2)
    elif fmt == "jsonl":
        with open(out_path, "w") as fh:
            for r in records:
                fh.write(
                    json.dumps(
                        {"id": r["id"], "prompt": r["prompt"], "generated_sql": r["generated_sql"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        raise ValueError(f"unknown format: {fmt!r}")


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

    runner = build_agent_runner(args.config)
    records = collect(cases, runner)
    write_output(records, args.out, args.format)

    print(f"collected {len(records)} cases -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
