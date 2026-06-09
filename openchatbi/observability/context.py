"""Run-context propagation via contextvars.

These are populated once at the start of each CLI/API turn so that deep code
(e.g. ``execute_sql_node``) can attribute work to a user/request *without*
threading ``user_id`` through every function signature. ContextVars copy into
asyncio tasks and ``contextvars.copy_context()`` (used by LangGraph's sync
ToolNode / ``asyncio.to_thread`` boundaries), so trace continuity holds.
"""

from __future__ import annotations

from contextvars import ContextVar

current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)


def set_run_context(user_id: str | None, request_id: str | None) -> None:
    """Bind the current user/request ids for the active context."""
    current_user_id.set(user_id)
    current_request_id.set(request_id)


def get_run_context() -> tuple[str | None, str | None]:
    """Return ``(user_id, request_id)`` for the active context."""
    return current_user_id.get(), current_request_id.get()
