"""Restore chat history from the LangGraph checkpointer for sample UIs.

The agent persists full conversation state (``messages``) in the SQLite
checkpointer keyed by ``thread_id = "{user_id}-{session_id}"``. The sample UIs
only render messages held in browser/session memory, so a page refresh loses
the visible history even though the agent itself still has the context.

This module bridges that gap: it reads the persisted ``messages`` via the
LangGraph ``get_state`` API (storage-agnostic) and converts them into the
simplified message dicts the UIs render. Intermediate tool calls and the
UI-only streaming artifacts (thinking steps, charts) are intentionally
dropped - the restored view shows user prompts and final AI text replies.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from openchatbi.observability.tracing import build_run_config
from openchatbi.utils import log


def _extract_text(content: Any) -> str:
    """Extract plain text from a message ``content`` (str or content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain content blocks: {"type": "text", "text": "..."}.
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def messages_to_ui_history(messages: Any) -> list[dict]:
    """Convert checkpoint LangChain messages into UI history message dicts.

    Mapping rules (simplified view):
    - ``HumanMessage`` -> ``{"role": "user", "type": "text", "content": <text>}``
    - ``AIMessage`` with non-empty text ->
      ``{"role": "assistant", "type": "text", "content": <text>}``
    - ``ToolMessage`` / ``SystemMessage`` / tool-call-only ``AIMessage`` /
      empty-content messages are skipped.

    Original ordering is preserved. Empty/None input returns ``[]``.
    """
    if not messages:
        return []

    history: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = _extract_text(message.content)
            if text:
                history.append({"role": "user", "type": "text", "content": text})
        elif isinstance(message, AIMessage):
            text = _extract_text(message.content)
            # Skip pure tool-call turns (no user-facing text).
            if text:
                history.append({"role": "assistant", "type": "text", "content": text})
        # ToolMessage, SystemMessage and anything else are intentionally dropped.
    return history


def _ui_history_to_tuples(history: list[dict]) -> list[tuple[str, str]]:
    """Pair an ordered simplified UI history into ``(user, assistant)`` tuples.

    Pairing rules (tolerant of non-strict alternation):
    - A ``user`` message opens a new ``(user, "")`` slot.
    - An ``assistant`` message fills the last slot if its assistant side is empty,
      otherwise opens a new ``("", assistant)`` slot.
    """
    pairs: list[list[str]] = []
    for item in history:
        if item["role"] == "user":
            pairs.append([item["content"], ""])
        else:  # assistant
            if pairs and pairs[-1][1] == "":
                pairs[-1][1] = item["content"]
            else:
                pairs.append(["", item["content"]])
    return [(user, assistant) for user, assistant in pairs]


def messages_to_chat_tuples(messages: Any) -> list[tuple[str, str]]:
    """Convert checkpoint messages into Gradio ``type="tuples"`` chat pairs.

    Reuses :func:`messages_to_ui_history` as the single source of truth (same
    filtering/ordering as the Streamlit view), then pairs the ordered simplified
    messages into ``(user, assistant)`` tuples. Empty/None input returns ``[]``.
    """
    return _ui_history_to_tuples(messages_to_ui_history(messages))


async def load_session_history(graph: Any, user_id: str, session_id: str) -> list[dict]:
    """Load persisted history for ``{user_id}-{session_id}`` as UI message dicts.

    Returns ``[]`` when the thread has no checkpoint, the state has no messages,
    or reading fails (graceful degradation - never blocks the conversation).
    """
    try:
        config = build_run_config(user_id=user_id, session_id=session_id)
        state = await graph.aget_state(config)
        if state is None:
            return []
        values = getattr(state, "values", None) or {}
        messages = values.get("messages", [])
        return messages_to_ui_history(messages)
    except Exception as exc:  # pragma: no cover - defensive degradation
        log(f"Failed to load session history for {user_id}-{session_id}: {exc!r}")
        return []


async def load_session_history_tuples(graph: Any, user_id: str, session_id: str) -> list[tuple[str, str]]:
    """Load persisted history as Gradio ``(user, assistant)`` chat tuples.

    Reuses :func:`load_session_history` for reading + graceful degradation, then
    pairs the result into tuples. Returns ``[]`` on missing/empty/failed reads.
    """
    history = await load_session_history(graph, user_id, session_id)
    return _ui_history_to_tuples(history)
