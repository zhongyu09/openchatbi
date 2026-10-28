"""Tests for streaming event parsing."""

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from openchatbi.streaming import AgentStreamProcessor, StreamStep, StreamToken


class Overwrite:
    def __init__(self, value):
        self.value = value


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


def test_empty_visualization_update_emits_no_step() -> None:
    processor = AgentStreamProcessor()

    events = processor.process((), "updates", {"generate_visualization": {"visualization_dsl": {}}})

    assert events == []


def test_data_analysis_failure_response_streams_as_main_final_tokens() -> None:
    """Tokens from data_analysis_failure_response must stream as main final answer
    (not sub-agent thinking), and main-graph helper nodes must not emit generic sub-agent steps."""
    processor = AgentStreamProcessor()

    chunk = AIMessageChunk(content="Analysis failed due to timeout.")
    events = processor.process(
        (),
        "messages",
        (chunk, {"langgraph_node": "data_analysis_failure_response", "streaming_tokens": True}),
    )

    assert len(events) == 1
    assert events[0] == StreamToken(
        text="Analysis failed due to timeout.",
        level=0,
        label="main",
        is_final=True,
    )
    assert processor.final_response == "Analysis failed due to timeout."

    # Updates from main-graph helper nodes must not be misclassified as sub-agent generic steps.
    hint_msg = ToolMessage(content="hint", tool_call_id="t1", name="data_analysis", id="m1")
    assert processor.process((), "updates", {"data_analysis_retry_hint": {"messages": [hint_msg]}}) == []
    assert processor.process((), "updates", {"reset_turn_state": {"data_analysis_retries": 0}}) == []
    assert (
        processor.process(
            (),
            "updates",
            {"data_analysis_failure_response": {"messages": [AIMessage(content="Analysis failed.")]}},
        )
        == []
    )
