"""Tests for streaming event parsing."""

from langchain_core.messages import ToolMessage

from openchatbi.streaming import AgentStreamProcessor, StreamStep


def test_use_tool_error_emits_tool_error_step() -> None:
    processor = AgentStreamProcessor()
    tool_error = ToolMessage(
        content=(
            "Error invoking tool 'text2sql' with kwargs {...} with error:\n"
            " context: Input should be a valid string\n"
            " Please fix the error and try again."
        ),
        tool_call_id="tool_call_1",
        name="text2sql",
        status="error",
    )

    events = processor.process((), "updates", {"use_tool": {"messages": [tool_error]}})

    assert len(events) == 1
    step = events[0]
    assert isinstance(step, StreamStep)
    assert step.kind == "tool_error"
    assert "Tool `text2sql` failed" in step.text
    assert "Input should be a valid string" in step.text
