"""Tracing provider integration (Langfuse v3 self-hosted, LangSmith fallback).

Credentials are read from environment / .env only (never from config files /
git). When tracing is disabled or the provider lib is missing, returns ``[]``
so the agent runs identically to today (zero regression).
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from openchatbi.observability.callbacks import ToolAuditCallback
from openchatbi.observability.context import set_run_context
from openchatbi.utils import log


def _resolve_observability_cfg() -> Any:
    try:
        from openchatbi import config as _cfg

        return getattr(_cfg.get(), "observability", None)
    except Exception:
        return None


def get_tracing_callbacks(enabled: bool | None = None, provider: str | None = None) -> list[BaseCallbackHandler]:
    """Build provider tracing callbacks; ``[]`` when disabled / unavailable."""
    obs = _resolve_observability_cfg()
    if enabled is None:
        enabled = bool(getattr(getattr(obs, "tracing", None), "enabled", False))
    if not enabled:
        return []
    if provider is None:
        provider = getattr(getattr(obs, "tracing", None), "provider", None) or "langfuse"

    if provider == "langfuse":
        try:
            from langfuse.langchain import CallbackHandler  # Langfuse v3 path

            # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
            return [CallbackHandler()]
        except Exception as exc:
            log(f"Langfuse tracing unavailable: {exc!r}")
            return []
    if provider == "langsmith":
        try:
            from langchain_core.tracers import LangChainTracer

            if not os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
                return []
            return [LangChainTracer()]
        except Exception as exc:
            log(f"LangSmith tracing unavailable: {exc!r}")
            return []
    return []


def build_run_config(
    user_id: str,
    session_id: str,
    request_id: str | None = None,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a LangGraph run config: configurable ids + tracing/audit callbacks + metadata.

    Also sets the run-context contextvars so deep code (execute_sql_node) can
    attribute work without signature threading.
    """
    set_run_context(user_id, request_id or f"{user_id}-{session_id}")

    cfg: dict[str, Any] = deepcopy(base) if base else {}
    configurable = dict(cfg.get("configurable") or {})
    configurable.setdefault("thread_id", f"{user_id}-{session_id}")
    configurable["user_id"] = user_id
    cfg["configurable"] = configurable

    callbacks: list[BaseCallbackHandler] = list(cfg.get("callbacks") or [])
    callbacks.append(ToolAuditCallback())
    callbacks.extend(get_tracing_callbacks())
    cfg["callbacks"] = callbacks

    metadata = dict(cfg.get("metadata") or {})
    metadata.update({"user_id": user_id, "session_id": session_id, "request_id": request_id})
    cfg["metadata"] = metadata

    cfg.setdefault("run_name", f"openchatbi:{user_id}:{session_id}")
    return cfg
