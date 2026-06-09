"""Structured audit logging for SQL executions and tool calls.

Never logs result bodies — only ``row_count`` and a short result preview. SQL
literals and tool-arg values are masked by default so PII / secrets never reach
the audit sink. Writes JSON lines to the ``openchatbi.audit`` logger.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_audit_logger = logging.getLogger("openchatbi.audit")

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_NUMBER_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def mask_sql(sql: str) -> str:
    """Redact string and numeric literals from SQL, preserving structure."""
    if not sql:
        return sql
    masked = _STRING_LITERAL.sub("'?'", sql)
    masked = _NUMBER_LITERAL.sub("?", masked)
    return masked


def mask_args(d: dict) -> dict:
    """Redact values of a tool-arg dict, keeping keys for traceability."""
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[k] = "<redacted>" if isinstance(v, str) and v else ("<redacted>" if v else v)
    return out


class AuditLogger:
    """Emits one structured JSON line per audited event."""

    def __init__(self, mask_literals: bool = True) -> None:
        self._mask = mask_literals

    def log_sql_exec(
        self,
        sql: str,
        dialect: str,
        row_count: int | None,
        duration_ms: float,
        status: str,
        user_id: str | None,
        error: str | None = None,
    ) -> None:
        payload = {
            "event": "sql_exec",
            "sql": mask_sql(sql) if self._mask else sql,
            "dialect": dialect,
            "row_count": row_count,
            "duration_ms": round(duration_ms, 2),
            "status": status,
            "user_id": user_id,
            "error": error,
        }
        _audit_logger.info(json.dumps(payload, ensure_ascii=False, default=str))

    def log_tool_call(
        self,
        tool: str,
        args: dict,
        result_preview: str,
        duration_ms: float,
        status: str,
        user_id: str | None,
    ) -> None:
        payload = {
            "event": "tool_call",
            "tool": tool,
            "args": mask_args(args) if self._mask else (args or {}),
            "result_preview": (result_preview or "")[:300],
            "duration_ms": round(duration_ms, 2),
            "status": status,
            "user_id": user_id,
        }
        _audit_logger.info(json.dumps(payload, ensure_ascii=False, default=str))
