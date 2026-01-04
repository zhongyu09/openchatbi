import json
import sys
from itertools import count
from typing import Any
from unittest.mock import MagicMock

import builtins
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from openchatbi import config
import openchatbi.agent_graph as agent_graph


_CALL_COUNTER = count(1)
_ORIG_PRINT = builtins.print


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Suppress stdout prints so JSONL stays clean; allow stderr."""
    target = kwargs.get("file")
    if target is None or target is sys.stdout:
        return
    _ORIG_PRINT(*args, **kwargs)


builtins.print = _safe_print


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


def _stub_llm_call(chat_model: Any, messages: list[Any], **_kwargs: Any) -> AIMessage:
    tool_seen = any(isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool" for msg in messages)
    if tool_seen:
        return AIMessage(content="Here is a deterministic summary based on the tool result.", tool_calls=[])

    user_text = _last_user_text(messages)
    tool_args = {
        "reasoning": "Look up relevant knowledge",
        "query_list": [user_text],
        "knowledge_bases": ["columns"],
        "with_table_list": False,
    }
    return AIMessage(
        content="Searching knowledge base.",
        tool_calls=[{"name": "search_knowledge", "args": tool_args, "id": "call_1"}],
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
    agent_graph.check_forecast_service_health = lambda: False
    agent_graph.call_llm_chat_model_with_retry = _stub_llm_call


def _bootstrap_config() -> None:
    config.set(
        {
            "default_llm": MagicMock(),
            "data_warehouse_config": {},
            "catalog_store": {"store_type": "file_system", "auto_load": False},
        }
    )


def main() -> int:
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
