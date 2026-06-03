#!/usr/bin/env python
"""Headless command-line runner for end-to-end testing of the OpenChatBI agent.

This drives the agent graph **in-process** (no Streamlit / HTTP needed) and
streams the same intermediate steps the Streamlit UI shows — selected tables,
generated SQL, SQL execution, visualizations, sub-agent thinking and tool calls
— by reusing :mod:`openchatbi.streaming`.

Examples
--------
Single question (sync graph, human-readable streaming output)::

    CONFIG_FILE=example/config.yaml python run_cli.py "How many users signed up last week?"

Interactive REPL (keeps the same thread/session, supports ask-human resume)::

    CONFIG_FILE=example/config.yaml python run_cli.py

Machine-readable NDJSON events (one JSON object per line, for automated E2E
assertions / piping into a test harness)::

    python run_cli.py --json "show me revenue by region" > events.ndjson

Use the async graph and disable streaming (just print the final answer)::

    python run_cli.py --async --no-stream "..."
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys

from openchatbi.streaming import (
    AgentStreamProcessor,
    StreamInterrupt,
    StreamStep,
    StreamToken,
    extract_final_answer,
)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_sync_graph(provider: str | None):
    """Build the synchronous agent graph with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from openchatbi import config
    from openchatbi.agent_graph import build_agent_graph_sync
    from openchatbi.tool.memory import get_sync_memory_store

    return build_agent_graph_sync(
        config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=get_sync_memory_store(),
        llm_provider=provider,
    )


async def build_async_graph(provider: str | None):
    """Build the asynchronous agent graph with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from openchatbi import config
    from openchatbi.agent_graph import build_agent_graph_async
    from openchatbi.tool.memory import get_async_memory_store

    return await build_agent_graph_async(
        config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=await get_async_memory_store(),
        llm_provider=provider,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class _Color:
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


class CliRenderer:
    """Render stream events to the terminal (human) or as NDJSON (machine)."""

    def __init__(self, as_json: bool = False, color: bool = True) -> None:
        self.as_json = as_json
        self.color = color and not as_json
        self._step_no = 0
        # Tracks which (level, label, is_final) layer the current token run
        # belongs to, so consecutive tokens share one header.
        self._token_layer: tuple | None = None
        self._on_token_line = False

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{_Color.RESET}" if self.color else text

    def _emit_json(self, event) -> None:
        if isinstance(event, StreamStep):
            payload = {"type": "step", "kind": event.kind, "level": event.level, "label": event.label, "text": event.text}
        elif isinstance(event, StreamToken):
            payload = {"type": "token", "level": event.level, "label": event.label, "is_final": event.is_final, "text": event.text}
        else:  # StreamInterrupt
            payload = {"type": "interrupt", "text": event.text, "buttons": event.buttons}
        print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)

    def render(self, event) -> None:
        if self.as_json:
            self._emit_json(event)
            return

        if isinstance(event, StreamToken):
            layer = (event.level, event.label, event.is_final)
            if self._token_layer != layer:
                self._end_token_line()
                if event.is_final:
                    header = self._c("\n🤖 AI Response: ", _Color.BOLD + _Color.GREEN)
                else:
                    indent = "  " * event.level
                    header = self._c(f"\n{indent}💭 {event.label} 思考: ", _Color.DIM)
                sys.stdout.write(header)
                self._token_layer = layer
            sys.stdout.write(event.text)
            sys.stdout.flush()
            self._on_token_line = True

        elif isinstance(event, StreamStep):
            self._end_token_line()
            self._token_layer = None
            if event.level == 0:
                self._step_no += 1
                prefix = self._c(f"Step {self._step_no}: ", _Color.BOLD + _Color.CYAN)
                print(f"\n{prefix}{event.text}")
            else:
                indent = "  " * event.level
                tag = self._c(f"↳ [{event.label}] ", _Color.DIM)
                print(f"\n{indent}{tag}{event.text}")

        elif isinstance(event, StreamInterrupt):
            self._end_token_line()
            self._token_layer = None
            box = self._c("⏸  Interrupt (input required):", _Color.BOLD + _Color.YELLOW)
            print(f"\n{box} {event.text}")
            if event.buttons:
                print(self._c(f"   options: {event.buttons}", _Color.DIM))

    def _end_token_line(self) -> None:
        if self._on_token_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._on_token_line = False

    def final_answer(self, text: str) -> None:
        if self.as_json:
            if text:
                print(json.dumps({"type": "final_answer", "text": text}, ensure_ascii=False), flush=True)
            return
        self._end_token_line()
        if text:
            banner = self._c("═" * 60, _Color.DIM)
            label = self._c("FINAL ANSWER", _Color.BOLD + _Color.GREEN)
            print(f"\n{banner}\n{label}\n{banner}\n{text}\n")


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# Turn execution (sync + async)
# --------------------------------------------------------------------------- #
def run_turn_sync(graph, stream_input, config, renderer: CliRenderer, stream: bool) -> bool:
    """Run one turn synchronously. Returns True if the graph is now interrupted."""
    processor = AgentStreamProcessor()
    if stream:
        for namespace, event_type, event_value in graph.stream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                renderer.render(event)
    else:
        graph.invoke(stream_input, config=config)

    state = graph.get_state(config)
    return _handle_state(state, processor, renderer)


async def run_turn_async(graph, stream_input, config, renderer: CliRenderer, stream: bool) -> bool:
    """Run one turn asynchronously. Returns True if the graph is now interrupted."""
    processor = AgentStreamProcessor()
    if stream:
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                renderer.render(event)
    else:
        await graph.ainvoke(stream_input, config=config)

    state = await graph.aget_state(config)
    return _handle_state(state, processor, renderer)


def _handle_state(state, processor: AgentStreamProcessor, renderer: CliRenderer) -> bool:
    """Render interrupt or final answer from the terminal state."""
    if state.interrupts:
        value = state.interrupts[0].value or {}
        renderer.render(StreamInterrupt(text=value.get("text", ""), buttons=value.get("buttons", []) or []))
        return True

    final = extract_final_answer(processor.final_response)
    if not final:
        # Fall back to the last message content when no streamed answer captured
        # (e.g. --no-stream, or a non-streaming final node).
        values = getattr(state, "values", None) or {}
        messages = values.get("messages") or []
        if messages:
            final = str(getattr(messages[-1], "content", "") or "")
    renderer.final_answer(final)
    return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _initial_input(message: str):
    return {"messages": [{"role": "user", "content": message}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless CLI to run the OpenChatBI agent end-to-end (streams intermediate steps)."
    )
    parser.add_argument("question", nargs="?", help="Question to ask. Omit to start an interactive REPL.")
    parser.add_argument("--user-id", default="cli", help="User id (default: cli).")
    parser.add_argument("--session-id", default="cli", help="Session id (default: cli).")
    parser.add_argument("--provider", default=None, help="LLM provider name (optional).")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use the async graph.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable step streaming; print only the final answer.")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit NDJSON events instead of human-readable output.")
    parser.add_argument("--no-color", dest="color", action="store_false", help="Disable ANSI colors.")
    args = parser.parse_args(argv)

    renderer = CliRenderer(as_json=args.as_json, color=args.color)
    config = {"configurable": {"thread_id": f"{args.user_id}-{args.session_id}", "user_id": args.user_id}}

    if args.use_async:
        return asyncio.run(_run_async(args, config, renderer))
    return _run_sync(args, config, renderer)


def _iter_questions(first: str | None):
    """Yield questions: the CLI arg once, then stdin lines (interactive REPL)."""
    if first is not None:
        yield first
        return
    print("OpenChatBI CLI — type a question (Ctrl-D / 'exit' to quit).", file=sys.stderr)
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if line.lower() in {"exit", "quit"}:
            break
        if line:
            yield line


def _run_sync(args, config, renderer: CliRenderer) -> int:
    from langgraph.types import Command

    graph = build_sync_graph(args.provider)
    interrupted = False
    for question in _iter_questions(args.question):
        stream_input = Command(resume=question) if interrupted else _initial_input(question)
        interrupted = run_turn_sync(graph, stream_input, config, renderer, args.stream)
    return 0


async def _run_async(args, config, renderer: CliRenderer) -> int:
    from langgraph.types import Command

    graph = await build_async_graph(args.provider)
    interrupted = False
    for question in _iter_questions(args.question):
        stream_input = Command(resume=question) if interrupted else _initial_input(question)
        interrupted = await run_turn_async(graph, stream_input, config, renderer, args.stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
