"""Tests for the tool-audit callback handler."""

from uuid import uuid4

from openchatbi.observability.callbacks import ToolAuditCallback


def test_callback_logs_tool_call_on_end(monkeypatch) -> None:
    calls: list[dict] = []

    class _Capture:
        def log_tool_call(self, tool, args, result_preview, duration_ms, status, user_id):
            calls.append(
                {"tool": tool, "preview": result_preview, "status": status, "user_id": user_id}
            )

    cb = ToolAuditCallback(audit=_Capture())
    run_id = uuid4()
    # run_python_code has no config param → callback still attributes it.
    cb.on_tool_start(
        {"name": "run_python_code"}, "print(1)", run_id=run_id,
        inputs={"code": "print(1)"},
    )
    cb.on_tool_end("hello-world-result", run_id=run_id)

    assert len(calls) == 1
    assert calls[0]["tool"] == "run_python_code"
    assert "hello-world-result" in calls[0]["preview"]
    assert calls[0]["status"] == "success"


def test_callback_logs_error(monkeypatch) -> None:
    calls: list[str] = []

    class _Capture:
        def log_tool_call(self, tool, args, result_preview, duration_ms, status, user_id):
            calls.append(status)

    cb = ToolAuditCallback(audit=_Capture())
    run_id = uuid4()
    cb.on_tool_start({"name": "text2sql"}, "q", run_id=run_id)
    cb.on_tool_error(ValueError("boom"), run_id=run_id)
    assert calls == ["error"]
