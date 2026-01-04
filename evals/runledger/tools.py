from __future__ import annotations

from typing import Any

from openchatbi.tool.search_knowledge import search_knowledge


def _invoke_tool(tool, args: dict[str, Any]) -> Any:
    return tool.invoke(args)


def _search_knowledge(args: dict[str, Any]) -> Any:
    return _invoke_tool(search_knowledge, args)


TOOLS = {
    "search_knowledge": _search_knowledge,
}
