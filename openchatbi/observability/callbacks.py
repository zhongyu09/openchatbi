"""LangChain callback that audits every tool call.

Registered once via ``build_run_config`` → ``config['callbacks']`` so it covers
ALL tools (text2sql / data_analysis / search_knowledge / save_report / MCP /
sub-agents) including ``run_python_code`` which has no ``config`` param and so
cannot be covered by a decorator.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from openchatbi.observability.audit import AuditLogger
from openchatbi.observability.context import get_run_context


class ToolAuditCallback(BaseCallbackHandler):
    """Maps on_tool_start/end/error onto ``AuditLogger.log_tool_call``."""

    def __init__(self, audit: AuditLogger | None = None) -> None:
        self._audit = audit or AuditLogger()
        self._pending: dict[UUID, dict[str, Any]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "tool"
        self._pending[run_id] = {
            "name": name,
            "args": inputs or {"input": input_str},
            "start": time.time(),
        }

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="success", result_preview=str(output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="error", result_preview=repr(error))

    def _finish(self, run_id: UUID, status: str, result_preview: str) -> None:
        info = self._pending.pop(run_id, None)
        if info is None:
            return
        user_id, _ = get_run_context()
        duration_ms = (time.time() - info["start"]) * 1000.0
        self._audit.log_tool_call(
            tool=info["name"],
            args=info["args"],
            result_preview=result_preview,
            duration_ms=duration_ms,
            status=status,
            user_id=user_id,
        )
