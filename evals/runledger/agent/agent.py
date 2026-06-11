import builtins
import json
import sys
from itertools import count
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

import openchatbi.agent_graph as agent_graph
from openchatbi import config

_CALL_COUNTER = count(1)


_REAL_PRINT = builtins.print  # captured at import time, before any patching


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Suppress stdout prints so JSONL stays clean; allow stderr."""
    target = kwargs.get("file")
    if target is None or target is sys.stdout:
        return
    _REAL_PRINT(*args, **kwargs)


# NOTE: builtins.print is patched inside main() only, not at import time.
# Patching at module level breaks test isolation (capsys sees empty stdout).


class JsonlChannel:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def read(self) -> dict[str, Any] | None:
        while True:
            line = self._stream.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    @staticmethod
    def send(payload: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def _last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return "OpenChatBI"


def _runledger_tool_call(channel: JsonlChannel, name: str, args: dict[str, Any]) -> Any:
    call_id = f"c{next(_CALL_COUNTER)}"
    channel.send({"type": "tool_call", "name": name, "call_id": call_id, "args": args})
    while True:
        message = channel.read()
        if message is None:
            raise RuntimeError("Tool result missing")
        if message.get("type") != "tool_result":
            continue
        if message.get("call_id") != call_id:
            continue
        if message.get("ok"):
            return message.get("result")
        raise RuntimeError(message.get("error") or "Tool error")


class SearchKnowledgeInput(BaseModel):
    reasoning: str = Field(description="Reason for searching knowledge")
    query_list: list[str] = Field(description="Query terms")
    knowledge_bases: list[str] = Field(description="Knowledge bases to search")
    with_table_list: bool = Field(default=False, description="Include table list")


class ShowSchemaInput(BaseModel):
    reasoning: str = Field(description="Reason for showing schema")
    tables: list[str] = Field(description="Table names")


class Text2SQLInput(BaseModel):
    reasoning: str = Field(description="Reason for calling text2sql")
    context: str = Field(description="Full context for the SQL graph")


class RunPythonInput(BaseModel):
    reasoning: str = Field(description="Reason for running python code")
    code: str = Field(description="Python code to execute")


class SaveReportInput(BaseModel):
    content: str = Field(description="Report content")
    title: str = Field(description="Report title")
    file_format: str = Field(description="File extension")


def _build_tool_proxies(channel: JsonlChannel) -> dict[str, StructuredTool]:
    def search_knowledge(
        reasoning: str,
        query_list: list[str],
        knowledge_bases: list[str],
        with_table_list: bool = False,
    ) -> Any:
        return _runledger_tool_call(
            channel,
            "search_knowledge",
            {
                "reasoning": reasoning,
                "query_list": query_list,
                "knowledge_bases": knowledge_bases,
                "with_table_list": with_table_list,
            },
        )

    def show_schema(reasoning: str, tables: list[str]) -> Any:
        return _runledger_tool_call(
            channel,
            "show_schema",
            {"reasoning": reasoning, "tables": tables},
        )

    def text2sql(reasoning: str, context: str) -> Any:
        return _runledger_tool_call(
            channel,
            "text2sql",
            {"reasoning": reasoning, "context": context},
        )

    def run_python_code(reasoning: str, code: str) -> Any:
        return _runledger_tool_call(
            channel,
            "run_python_code",
            {"reasoning": reasoning, "code": code},
        )

    def save_report(content: str, title: str, file_format: str = "md") -> Any:
        return _runledger_tool_call(
            channel,
            "save_report",
            {"content": content, "title": title, "file_format": file_format},
        )

    return {
        "search_knowledge": StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description="RunLedger proxy for search_knowledge",
            args_schema=SearchKnowledgeInput,
        ),
        "show_schema": StructuredTool.from_function(
            func=show_schema,
            name="show_schema",
            description="RunLedger proxy for show_schema",
            args_schema=ShowSchemaInput,
        ),
        "text2sql": StructuredTool.from_function(
            func=text2sql,
            name="text2sql",
            description="RunLedger proxy for text2sql",
            args_schema=Text2SQLInput,
        ),
        "run_python_code": StructuredTool.from_function(
            func=run_python_code,
            name="run_python_code",
            description="RunLedger proxy for run_python_code",
            args_schema=RunPythonInput,
        ),
        "save_report": StructuredTool.from_function(
            func=save_report,
            name="save_report",
            description="RunLedger proxy for save_report",
            args_schema=SaveReportInput,
        ),
    }


# Each trajectory: list of turns. A turn is either a tool name (emit one tool_call)
# or None (emit a final text answer with no tool_calls). The driver advances by the
# count of ToolMessages already present, because the case-id is NOT in the JSONL
# protocol — the only stable key is the user prompt text.
_TRAJECTORIES: dict[str, list[str | None]] = {
    "OpenChatBI": ["search_knowledge", None],
    "How many orders were placed in 2024?": ["text2sql", None],
    "What is the total revenue by region?": ["text2sql", None],
    "What is the average order value per customer?": ["text2sql", None],
    "Show daily active users for the last 30 days": ["text2sql", None],
    "Join orders with customers and list top 10 spenders": ["text2sql", None],
    "Which products have orders but no shipments?": ["text2sql", None],
    "What were sales between 2024-01-01 and 2024-03-31?": ["text2sql", None],
    "Compare this month's revenue to last month": ["text2sql", None],
    "Detect anomalies in daily signup counts": ["text2sql", "run_python_code", None],
    "Plot the revenue trend for 2024": ["text2sql", "run_python_code", None],
    "What columns describe customer churn?": ["search_knowledge", None],
    "Explain the orders fact table": ["show_schema", None],
    "What does the metric DAU mean?": ["search_knowledge", None],
    "List the schema of the customers table": ["show_schema", None],
    "How many active users signed up last week?": ["text2sql", None],
    "Forecast next quarter revenue from history": ["text2sql", "run_python_code", None],
    "Break down conversion rate by channel": ["text2sql", None],
    "Generate a sales report for Q1 2024": ["search_knowledge", "text2sql", "save_report", None],
    "Summarize order volume and save it as a report": ["text2sql", "save_report", None],
}

# Default trajectory for any prompt not in the table (e.g. novel record-mode runs).
_DEFAULT_TRAJECTORY: list[str | None] = ["search_knowledge", None]

_TOOL_ARGS_BUILDERS = {
    "search_knowledge": lambda q: {
        "reasoning": "Look up relevant knowledge",
        "query_list": [q],
        "knowledge_bases": ["columns"],
        "with_table_list": False,
    },
    "show_schema": lambda q: {"reasoning": "Inspect schema", "tables": [q]},
    "text2sql": lambda q: {"reasoning": "Generate SQL", "context": q},
    "run_python_code": lambda q: {
        "reasoning": "Post-process result",
        "code": "result = df.describe()",
    },
    "save_report": lambda q: {
        "content": f"Report for: {q}",
        "title": q[:40].rstrip() or "report",
        "file_format": "md",
    },
}


def _tool_message_count(messages: list[Any]) -> int:
    return sum(
        1 for m in messages if isinstance(m, ToolMessage) or getattr(m, "type", None) == "tool"
    )


def _scripted_llm_call(chat_model: Any, messages: list[Any], **_kwargs: Any) -> AIMessage:
    user_text = _last_user_text(messages)
    trajectory = _TRAJECTORIES.get(user_text, _DEFAULT_TRAJECTORY)
    step = _tool_message_count(messages)
    if step >= len(trajectory) or trajectory[step] is None:
        return AIMessage(content="Here is a deterministic summary based on the tool result.", tool_calls=[])
    tool_name = trajectory[step]
    args = _TOOL_ARGS_BUILDERS[tool_name](user_text)
    call_id = f"call_{step + 1}"
    return AIMessage(
        content=f"Calling {tool_name}.",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
    )


def _configure_agent_graph(channel: JsonlChannel) -> None:
    tool_proxies = _build_tool_proxies(channel)

    agent_graph.search_knowledge = tool_proxies["search_knowledge"]
    agent_graph.show_schema = tool_proxies["show_schema"]
    agent_graph.run_python_code = tool_proxies["run_python_code"]
    agent_graph.save_report = tool_proxies["save_report"]
    agent_graph.get_sql_tools = lambda *_args, **_kwargs: tool_proxies["text2sql"]
    agent_graph.build_sql_graph = lambda *_args, **_kwargs: object()
    agent_graph.get_memory_tools = lambda *_args, **_kwargs: []
    agent_graph.create_mcp_tools_sync = lambda *_args, **_kwargs: []
    # Forecast health is now checked inside the data analysis sub-agent, not the
    # main agent graph; stub it there to keep the eval hermetic (no network).
    import openchatbi.analysis.agent as analysis_agent

    analysis_agent.check_forecast_service_health = lambda: False
    agent_graph.call_llm_chat_model_with_retry = _scripted_llm_call


def _bootstrap_config() -> None:
    config.set(
        {
            "default_llm": MagicMock(),
            "data_warehouse_config": {},
            "catalog_store": {"store_type": "file_system", "auto_load": False},
        }
    )


def main() -> int:
    # Suppress stdout prints inside main() so JSONL output stays clean.
    # This is intentionally scoped to main() and NOT done at import time so that
    # importing this module in tests does not pollute capsys/stdout capture.
    builtins.print = _safe_print

    channel = JsonlChannel(sys.stdin)
    message = channel.read()
    if not message or message.get("type") != "task_start":
        return 1

    _bootstrap_config()
    _configure_agent_graph(channel)

    prompt = ""
    payload = message.get("input", {})
    if isinstance(payload, dict):
        prompt = payload.get("prompt") or payload.get("question") or payload.get("query") or ""
    if not prompt:
        prompt = "OpenChatBI"

    graph = agent_graph.build_agent_graph_sync(
        catalog=config.get().catalog_store,
        checkpointer=MemorySaver(),
        memory_store=None,
        enable_context_management=False,
    )

    result = graph.invoke({"messages": [{"role": "user", "content": prompt}]})
    output = ""
    if isinstance(result, dict) and result.get("messages"):
        output = str(result["messages"][-1].content)

    channel.send(
        {
            "type": "final_output",
            "output": {"category": "bi", "reply": output or "Completed request."},
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
