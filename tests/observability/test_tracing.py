"""Tests for tracing callbacks + build_run_config."""

from openchatbi.observability.tracing import build_run_config, get_tracing_callbacks


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
    assert cfg["metadata"]["user_id"] == "alice"
    assert cfg["metadata"]["request_id"] == "req-1"
    assert cfg["run_name"]


def test_build_run_config_preserves_base() -> None:
    base = {"configurable": {"thread_id": "existing-tid", "extra": 1}, "recursion_limit": 50}
    cfg = build_run_config(user_id="bob", session_id="s2", base=base)
    # base values survive; thread_id from base is preserved if already set.
    assert cfg["recursion_limit"] == 50
    assert cfg["configurable"]["extra"] == 1
    assert cfg["configurable"]["user_id"] == "bob"
