from __future__ import annotations

from typing import Any

from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.save_report import save_report
from openchatbi.tool.search_knowledge import search_knowledge, search_schema, show_schema


def _invoke_tool(tool, args: dict[str, Any]) -> Any:
    return tool.invoke(args)


def _search_knowledge(args: dict[str, Any]) -> Any:
    return _invoke_tool(search_knowledge, args)


def _search_schema(args: dict[str, Any]) -> Any:
    return _invoke_tool(search_schema, args)


def _show_schema(args: dict[str, Any]) -> Any:
    return _invoke_tool(show_schema, args)


def _run_python_code(args: dict[str, Any]) -> Any:
    return _invoke_tool(run_python_code, args)


def _save_report(args: dict[str, Any]) -> Any:
    return _invoke_tool(save_report, args)


def _text2sql(args: dict[str, Any]) -> Any:
    # text2sql is graph-built (get_sql_tools), not a module-level @tool callable.
    # In record mode the SQL sub-graph requires a live warehouse/LLM; the eval is
    # replay-only by default, so this proxy is a deterministic echo for recording.
    return {"sql": "", "data": "", "context": args.get("context", "")}


TOOLS = {
    "search_knowledge": _search_knowledge,
    "search_schema": _search_schema,
    "show_schema": _show_schema,
    "text2sql": _text2sql,
    "run_python_code": _run_python_code,
    "save_report": _save_report,
}
