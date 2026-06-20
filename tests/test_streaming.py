"""Tests for streaming event parsing."""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Overwrite

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


def test_use_tool_error_accepts_overwrite_messages() -> None:
    processor = AgentStreamProcessor()
    tool_error = ToolMessage(
        content="Error invoking tool 'search_schema' with error:\n bad schema",
        tool_call_id="tool_call_1",
        name="search_schema",
        status="error",
    )

    events = processor.process((), "updates", {"use_tool": {"messages": Overwrite(value=[tool_error])}})

    assert len(events) == 1
    assert isinstance(events[0], StreamStep)
    assert events[0].kind == "tool_error"
    assert "Tool `search_schema` failed" in events[0].text


def test_generic_subagent_node_accepts_overwrite_messages() -> None:
    processor = AgentStreamProcessor()
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_schema",
                "args": {"reasoning": "Find order tables"},
                "id": "call_1",
            }
        ],
    )

    events = processor.process((), "updates", {"model": {"messages": Overwrite(value=[ai_message])}})

    assert len(events) == 1
    assert isinstance(events[0], StreamStep)
    assert events[0].kind == "generic"
    assert "Using tool: `search_schema`" in events[0].text
