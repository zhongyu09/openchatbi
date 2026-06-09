"""UI-agnostic parsing of agent-graph stream events into human-readable steps.

The Streamlit UI (`sample_ui/streamlit_ui.py`) historically owned the logic that
turns the raw ``graph.astream(..., stream_mode=["updates", "messages"],
subgraphs=True)`` triples into readable steps (selected tables, generated SQL,
sub-agent thinking, tool calls, ...). That logic is reused here by the CLI and
the HTTP API so all three surfaces stay in sync.

The parser is intentionally free of any rendering / framework dependency: it
emits structured :class:`StreamEvent` objects and lets each consumer decide how
to display (terminal, NDJSON, Streamlit markdown, ...). Building a Plotly figure
from a visualization step is left to the consumer (it is UI-specific); the
``visualization`` step instead carries the raw DSL plus the latest CSV data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from openchatbi.observability.pricing import estimate_cost
from openchatbi.utils import get_text_from_message_chunk

# Node names that belong to the text2sql SQL subgraph. Their intermediate token
# stream is intentionally not surfaced as raw "thinking" because the dedicated
# `updates` handlers below already render them (SQL, tables, etc.).
SQL_SUBGRAPH_NODES = {
    "search_knowledge",
    "ask_human",
    "information_extraction",
    "table_selection",
    "generate_sql",
    "execute_sql",
    "regenerate_sql",
    "generate_visualization",
}

# Node names of the top-level agent graph (openchatbi). Anything else that
# bubbles up at depth 0 is the data analysis sub-agent: because it is invoked
# with a reset checkpoint namespace (see analysis/agent._build_sub_agent_config),
# the deepagents `model`/`tools`/middleware nodes are flattened onto depth 0
# instead of appearing as a nested subgraph.
MAIN_GRAPH_NODES = {"llm_node", "use_tool", "ask_human"}

# Display label used for the (flattened) data analysis sub-agent layer.
SUBAGENT_LABEL = "data_analysis"


@dataclass
class StreamToken:
    """A streamed LLM token (partial assistant text)."""

    text: str
    level: int  # 0 = main agent, >0 = nested / sub-agent
    label: str  # "main", "data_analysis", or a namespace label
    is_final: bool  # True for the main agent's (intermediate/final) answer tokens


@dataclass
class StreamStep:
    """A completed intermediate step from a graph node (the `updates` events)."""

    text: str  # human-readable, possibly markdown (e.g. fenced SQL)
    level: int  # 0 = main agent, >0 = nested / sub-agent
    label: str  # "main", "data_analysis", or a namespace label
    kind: str  # "tool" | "sub_agent" | "rewrite" | "tables" | "sql" |
    # "execute_sql" | "visualization" | "tool_error" | "generic"
    data: dict[str, Any] = field(default_factory=dict)  # structured payload


@dataclass
class StreamInterrupt:
    """An ask-human interrupt raised by the graph (requires a resume)."""

    text: str
    buttons: list[Any] = field(default_factory=list)


@dataclass
class StreamUsage:
    """Per-turn token / cost rollup, surfaced once at the end of a turn."""

    turn_tokens: int
    turn_cost_usd: float
    by_model: dict[str, int] = field(default_factory=dict)


StreamEvent = StreamToken | StreamStep | StreamInterrupt | StreamUsage


def format_namespace_label(namespace: tuple) -> str:
    """Turn an astream subgraph namespace into a readable layer label.

    A namespace looks like ``("use_tool:abc", "tools:xyz", ...)``; we keep the
    node-name part of each segment so users can see which layer a step belongs
    to (e.g. ``use_tool › tools``).
    """
    if not namespace:
        return "main"
    parts = []
    for seg in namespace:
        name = str(seg).split(":")[0]
        # Skip the bare numeric index layers LangGraph inserts for subgraphs
        # invoked without an explicit config (e.g. text2sql's sql_graph).
        if name.isdigit():
            continue
        parts.append(name)
    return " › ".join(parts) if parts else "subtask"


def preview_text(text: object, limit: int = 300) -> str:
    """Collapse whitespace and truncate long content for compact previews."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + " …(truncated)"


def _extract_tool_error_message(content: object) -> str:
    """Extract concise error details from a tool error payload."""
    text = str(content)
    marker = "with error:"
    if marker in text:
        error_detail = text.split(marker, 1)[1]
        # Keep only the actionable part and drop generic tail guidance.
        text = error_detail.split("Please fix", 1)[0].strip() or error_detail
    return preview_text(text, 300)


def _describe_generic_node(node_output: dict) -> list[str]:
    """Describe tool calls / tool results from an arbitrary sub-agent node.

    Used as a fallback for nodes without a dedicated handler (notably the
    deepagents data-analysis sub-agent's ``model``/``tools`` nodes). Plain
    assistant text is skipped here because it is already shown via the token
    stream.
    """
    descriptions: list[str] = []
    for message in node_output.get("messages") or []:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(message, AIMessage) and tool_calls:
            for tool_call in tool_calls:
                args = tool_call.get("args") or {}
                rationale = ""
                for key in ("reasoning", "task", "question", "query", "goal"):
                    if args.get(key):
                        rationale = preview_text(args[key], 200)
                        break
                suffix = f"：{rationale}" if rationale else ""
                descriptions.append(f"🛠️ Using tool: `{tool_call.get('name', '?')}`{suffix}")
        elif isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or "tool"
            descriptions.append(f"📤 Tool `{tool_name}` result：{preview_text(message.content, 300)}")
    return descriptions


class AgentStreamProcessor:
    """Stateful parser that converts astream triples into :class:`StreamEvent`.

    Usage::

        processor = AgentStreamProcessor()
        async for namespace, event_type, event_value in graph.astream(...):
            for event in processor.process(namespace, event_type, event_value):
                ...  # render however you like

    The instance keeps a little state across events (the latest CSV produced by
    ``execute_sql`` so a later ``generate_visualization`` step can carry it, and
    an accumulator of the main agent's answer tokens for final-answer extraction).
    """

    def __init__(self) -> None:
        self._data_csv: Any = None
        self.final_response: str = ""
        self.turn_usage: StreamUsage = StreamUsage(turn_tokens=0, turn_cost_usd=0.0, by_model={})

    def process(self, namespace: tuple, event_type: str, event_value: Any) -> list[StreamEvent]:
        """Parse a single astream triple into zero or more stream events."""
        depth = len(namespace) if namespace else 0
        layer_label = format_namespace_label(namespace)

        if event_type == "messages":
            return list(self._process_message(namespace, depth, layer_label, event_value))
        return list(self._process_updates(depth, layer_label, event_value))

    def emit_turn_usage(self) -> StreamUsage | None:
        """Return the accumulated per-turn usage, or None if nothing was recorded."""
        if self.turn_usage.turn_tokens <= 0:
            return None
        return self.turn_usage

    def _process_message(
        self, namespace: tuple, depth: int, layer_label: str, event_value: Any
    ) -> Iterator[StreamEvent]:
        chunk, metadata = event_value[0], event_value[1]
        node = metadata.get("langgraph_node")

        # Fold usage_metadata from this chunk into the per-turn accumulator
        # before the early-return guard so that the final (often empty-text)
        # chunk is still captured.
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            total = int(usage.get("total_tokens", 0) or 0)
            model_name = (getattr(chunk, "response_metadata", None) or {}).get("model_name", "") or "unknown"
            if total:
                self.turn_usage.turn_tokens += total
                self.turn_usage.by_model[model_name] = self.turn_usage.by_model.get(model_name, 0) + total
                self.turn_usage.turn_cost_usd += estimate_cost(
                    model_name,
                    int(usage.get("input_tokens", 0) or 0),
                    int(usage.get("output_tokens", 0) or 0),
                )

        token = get_text_from_message_chunk(chunk) if isinstance(chunk, AIMessageChunk | AIMessage) else ""
        if not token:
            return

        if depth == 0 and node == "llm_node" and metadata.get("streaming_tokens", False):
            # Main agent streaming its (intermediate/final) answer.
            self.final_response += token
            yield StreamToken(text=token, level=0, label="main", is_final=True)
        elif node in SQL_SUBGRAPH_NODES:
            # SQL subgraph internal LLM tokens: rendered as steps via `updates`.
            return
        elif depth == 0 and node in MAIN_GRAPH_NODES:
            # Other main-graph nodes don't carry a useful thinking trace.
            return
        else:
            # Sub-agent (data analysis) LLM thinking tokens. They bubble up
            # nested (depth>0) or flattened onto depth 0 (reset checkpoint ns).
            level = depth if depth > 0 else 1
            label = layer_label if depth > 0 else SUBAGENT_LABEL
            yield StreamToken(text=token, level=level, label=label, is_final=False)

    def _process_updates(self, depth: int, layer_label: str, event_value: Any) -> Iterator[StreamEvent]:
        for node_name, node_output in event_value.items():
            if not isinstance(node_output, dict):
                continue

            is_main_node = depth == 0 and node_name in MAIN_GRAPH_NODES
            if is_main_node:
                level, label = 0, "main"
            elif depth == 0:
                # deepagents sub-agent node flattened onto depth 0.
                level, label = 1, SUBAGENT_LABEL
            else:
                level, label = depth, layer_label

            matched = True
            kind = "generic"
            desc: str | None = None
            data: dict[str, Any] = {}
            extra_steps: list[StreamStep] = []

            if node_name == "llm_node":
                message_obj = (node_output.get("messages") or [None])[0]
                if isinstance(message_obj, AIMessage) and message_obj.tool_calls:
                    sub_agents = [
                        t["name"] for t in message_obj.tool_calls if t["name"] in ("data_analysis", "text2sql")
                    ]
                    normal_tools = [t["name"] for t in message_obj.tool_calls if t["name"] not in sub_agents]
                    if normal_tools:
                        desc = f"🛠️ Using tools: {', '.join(normal_tools)}"
                        kind = "tool"
                        data = {"tools": normal_tools}
                    if sub_agents:
                        desc = f"🛠️ Running sub agent: {', '.join(sub_agents)}"
                        kind = "sub_agent"
                        data = {"sub_agents": sub_agents}

            elif node_name == "information_extraction":
                message_obj = (node_output.get("messages") or [None])[0]
                if message_obj and getattr(message_obj, "tool_calls", None):
                    tool_name = message_obj.tool_calls[0]["name"]
                    desc = f"🛠️ Using tool: {tool_name}"
                    kind = "tool"
                    data = {"tools": [tool_name]}
                else:
                    rewrite_q = node_output.get("rewrite_question")
                    if rewrite_q:
                        desc = f"📝 Rewriting question: {rewrite_q}"
                        kind = "rewrite"
                        data = {"rewrite_question": rewrite_q}

            elif node_name == "table_selection":
                tables = node_output.get("tables")
                if tables:
                    desc = f"🗂️ Selected tables: {tables}"
                    kind = "tables"
                    data = {"tables": tables}

            elif node_name == "generate_sql":
                sql = node_output.get("sql")
                if sql:
                    desc = f"💾 Generated SQL:\n```sql\n{sql}\n```"
                    kind = "sql"
                    data = {"sql": sql}

            elif node_name == "execute_sql":
                desc = "⚡ Executing SQL query..."
                kind = "execute_sql"
                self._data_csv = node_output.get("data")
                data = {"data": self._data_csv}

            elif node_name == "regenerate_sql":
                sql = node_output.get("sql")
                if sql:
                    desc = f"🔄 Regenerated SQL:\n```sql\n{sql}\n```"
                    kind = "sql"
                    data = {"sql": sql}

            elif node_name == "generate_visualization":
                visualization_dsl = node_output.get("visualization_dsl")
                if visualization_dsl and "error" not in visualization_dsl:
                    desc = "📊 Generated visualization"
                    kind = "visualization"
                    data = {"visualization_dsl": visualization_dsl, "data": self._data_csv}
            elif node_name == "use_tool":
                for message in node_output.get("messages") or []:
                    if not isinstance(message, ToolMessage):
                        continue
                    if getattr(message, "status", None) != "error":
                        continue
                    tool_name = getattr(message, "name", None) or "tool"
                    error_preview = _extract_tool_error_message(message.content)
                    extra_steps.append(
                        StreamStep(
                            text=f"❌ Tool `{tool_name}` failed: {error_preview}",
                            level=level,
                            label=label,
                            kind="tool_error",
                            data={"tool": tool_name, "error": str(message.content)},
                        )
                    )
            else:
                matched = False

            yield from extra_steps

            if desc:
                yield StreamStep(text=desc, level=level, label=label, kind=kind, data=data)

            # Generic fallback for sub-agent nodes (deepagents `model`/`tools`).
            if not matched and not is_main_node:
                for generic_desc in _describe_generic_node(node_output):
                    yield StreamStep(text=generic_desc, level=level, label=label, kind="generic")


def extract_final_answer(final_response: str) -> str:
    """Extract the trailing final answer (text after the last tool-usage marker).

    Mirrors the heuristic the Streamlit UI uses to split the final answer from
    the intermediate "Using tool" chatter in the accumulated response text.
    """
    if not final_response:
        return ""
    lines = final_response.split("\n")
    final_answer_lines: list[str] = []
    collecting = False
    for line in reversed(lines):
        if "Use tool:" in line or "Using tools:" in line or "Using tool:" in line:
            break
        final_answer_lines.append(line)
        collecting = True
    if collecting and final_answer_lines:
        final_answer_lines.reverse()
        return "\n".join(final_answer_lines).strip()
    return ""
