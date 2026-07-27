"""Tests for tracing callbacks + build_run_config."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from openchatbi.observability.tracing import build_run_config, get_tracing_callbacks


def _metadata_echo_graph():
    def capture_metadata(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return {"seen_metadata": dict(config.get("metadata") or {})}

    graph = StateGraph(dict)
    graph.add_node("capture_metadata", capture_metadata)
    graph.add_edge(START, "capture_metadata")
    graph.add_edge("capture_metadata", END)
    return graph.compile()


def test_get_tracing_callbacks_disabled_returns_empty(monkeypatch) -> None:
    # No provider configured / disabled → empty list (zero-regression default).
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_tracing_callbacks(enabled=False) == []


def test_build_run_config_shape() -> None:
    cfg = build_run_config(user_id="alice", session_id="sess-1", request_id="req-1")
    assert cfg["configurable"]["thread_id"] == "alice-sess-1"
    assert cfg["configurable"]["user_id"] == "alice"
    assert isinstance(cfg["callbacks"], list)
    assert cfg["metadata"]["langfuse_user_id"] == "alice"
    assert cfg["metadata"]["langfuse_session_id"] == "sess-1"
    assert cfg["metadata"]["langfuse_trace_name"] == cfg["run_name"]
    assert cfg["metadata"]["request_id"] == "req-1"
    assert "user_id" not in cfg["metadata"]
    assert "session_id" not in cfg["metadata"]
    assert cfg["run_name"]


def test_build_run_config_preserves_base() -> None:
    base = {"configurable": {"thread_id": "existing-tid", "extra": 1}, "recursion_limit": 50}
    cfg = build_run_config(user_id="bob", session_id="s2", base=base)
    # base values survive; thread_id from base is preserved if already set.
    assert cfg["recursion_limit"] == 50
    assert cfg["configurable"]["extra"] == 1
    assert cfg["configurable"]["user_id"] == "bob"


def test_langfuse_observability_extra_requires_v4_fast_preview_sdk() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["observability"]

    assert "langfuse>=4.7,<5" in deps


def test_langfuse_correlation_fields_are_promoted_metadata() -> None:
    cfg = build_run_config(user_id="alice", session_id="sess-1", request_id="req-1")

    # Langfuse CallbackHandler promotes these reserved metadata keys to trace fields.
    assert cfg["metadata"]["langfuse_user_id"] == "alice"
    assert cfg["metadata"]["langfuse_session_id"] == "sess-1"
    assert cfg["metadata"]["langfuse_trace_name"] == "openchatbi:alice:sess-1"
    assert cfg["metadata"]["request_id"] == "req-1"
    assert "user_id" not in cfg["metadata"]
    assert "session_id" not in cfg["metadata"]


def test_sync_langgraph_execution_receives_langfuse_metadata() -> None:
    graph = _metadata_echo_graph()
    cfg = build_run_config(user_id="alice", session_id="sess-1", request_id="req-1")

    updates = list(graph.stream({"input": "hi"}, config=cfg, stream_mode="updates"))
    seen_metadata = updates[0]["capture_metadata"]["seen_metadata"]

    assert seen_metadata["langfuse_user_id"] == "alice"
    assert seen_metadata["langfuse_session_id"] == "sess-1"
    assert seen_metadata["langfuse_trace_name"] == cfg["run_name"]
    assert seen_metadata["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_async_langgraph_execution_receives_langfuse_metadata() -> None:
    graph = _metadata_echo_graph()
    cfg = build_run_config(user_id="alice", session_id="sess-1", request_id="req-1")

    updates = [update async for update in graph.astream({"input": "hi"}, config=cfg, stream_mode="updates")]
    seen_metadata = updates[0]["capture_metadata"]["seen_metadata"]

    assert seen_metadata["langfuse_user_id"] == "alice"
    assert seen_metadata["langfuse_session_id"] == "sess-1"
    assert seen_metadata["langfuse_trace_name"] == cfg["run_name"]
    assert seen_metadata["request_id"] == "req-1"
