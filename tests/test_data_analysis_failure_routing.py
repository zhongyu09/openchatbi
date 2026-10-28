"""Tests for data_analysis failure routing and surfacing in the main graph."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from openchatbi.agent_graph import (
    _latest_data_analysis_failure,
    _parse_data_analysis_failure,
    _tool_succeeded,
    data_analysis_retry_hint,
    make_data_analysis_failure_response,
    reset_turn_state,
    route_after_tool,
)
from openchatbi.analysis.agent import (
    DATA_ANALYSIS_ERROR_PREFIX,
    DATA_ANALYSIS_FAILED_MARKER,
    _format_data_analysis_error,
)
from openchatbi.graph_state import AgentState


def _make_failure_content(tag: str = "[PERMANENT]", body: str = "boom") -> str:
    return f"{DATA_ANALYSIS_FAILED_MARKER}{tag} {DATA_ANALYSIS_ERROR_PREFIX} {body}"


def test_parse_data_analysis_failure_permanent():
    content = _make_failure_content("[PERMANENT]", "something went wrong")
    tag, body = _parse_data_analysis_failure(content)
    assert tag == "[PERMANENT]"
    assert body == "something went wrong"


def test_parse_data_analysis_failure_retryable():
    content = _make_failure_content("[RETRYABLE]", "too many steps")
    tag, body = _parse_data_analysis_failure(content)
    assert tag == "[RETRYABLE]"
    assert body == "too many steps"


def test_parse_data_analysis_failure_non_failure_returns_none():
    assert _parse_data_analysis_failure("Normal analysis result") is None
    assert _parse_data_analysis_failure("") is None


def test_latest_data_analysis_failure_finds_last_failure_in_turn():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(content="first result", tool_call_id="t1", name="data_analysis"),
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t2"}]),
            ToolMessage(content=_make_failure_content("[RETRYABLE]", "oops"), tool_call_id="t2", name="data_analysis"),
        ],
    )
    msg, tag, body = _latest_data_analysis_failure(state)
    assert tag == "[RETRYABLE]"
    assert body == "oops"


def test_route_after_tool_no_failure_goes_to_llm_node():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(content="All good", tool_call_id="t1", name="data_analysis"),
        ],
    )
    assert route_after_tool(state) == "llm_node"


def test_route_after_tool_retryable_goes_to_retry_hint():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_make_failure_content("[RETRYABLE]", "recursion"),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "data_analysis_retry_hint"


def test_route_after_tool_permanent_goes_to_failure_response():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_make_failure_content("[PERMANENT]", "bug"),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "data_analysis_failure_response"


def test_route_after_tool_exhausted_retries_goes_to_failure_response():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_make_failure_content("[RETRYABLE]", "again"),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=1,
    )
    assert route_after_tool(state) == "data_analysis_failure_response"


def test_data_analysis_retry_hint_increments_counter_and_adds_hint_message():
    state = AgentState(
        messages=[
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_format_data_analysis_error(RuntimeError("too many steps")),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    result = data_analysis_retry_hint(state)
    assert result["data_analysis_retries"] == 1
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)
    assert "retryable" in result["messages"][0].content


def test_make_data_analysis_failure_response_calls_llm_without_tools():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Sorry, analysis failed.")

    state = AgentState(
        messages=[
            HumanMessage(content="do analysis"),
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_format_data_analysis_error(RuntimeError("boom")),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )

    node = make_data_analysis_failure_response(mock_llm)
    result = node(state)

    # The failure-response node is terminal (goes to END) and must NOT touch the
    # retry counter; the counter is reset per-turn by reset_turn_state instead.
    assert "data_analysis_retries" not in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Sorry, analysis failed."
    # The failure-response node should invoke the raw LLM, not a tool-bound one.
    mock_llm.invoke.assert_called_once()
    assert not mock_llm.bind_tools.called


def test_reset_turn_state_resets_retry_counter():
    """reset_turn_state zeroes the retry budget for each new user invocation."""
    state = AgentState(messages=[HumanMessage(content="hi")], data_analysis_retries=3)
    result = reset_turn_state(state)
    assert result == {"data_analysis_retries": 0}


def test_route_after_tool_parallel_success_goes_to_llm_node():
    """When a sibling tool succeeded in the same batch as a failed data_analysis,
    route back to llm_node so the successful result is not dropped."""
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "text2sql", "args": {}, "id": "t1"},
                    {"name": "data_analysis", "args": {}, "id": "t2"},
                ],
            ),
            ToolMessage(content="SQL ok", tool_call_id="t1", name="text2sql"),
            ToolMessage(
                content=_make_failure_content("[PERMANENT]", "boom"),
                tool_call_id="t2",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "llm_node"


def test_route_after_tool_all_failed_batch_goes_to_failure_response():
    """If every tool in the batch is a data_analysis failure, the structural
    no-redo guarantee still applies (route to failure_response, not llm_node)."""
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "data_analysis", "args": {}, "id": "t1"},
                    {"name": "data_analysis", "args": {}, "id": "t2"},
                ],
            ),
            ToolMessage(
                content=_make_failure_content("[PERMANENT]", "boom1"),
                tool_call_id="t1",
                name="data_analysis",
            ),
            ToolMessage(
                content=_make_failure_content("[PERMANENT]", "boom2"),
                tool_call_id="t2",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "data_analysis_failure_response"


def test_tool_succeeded_treats_non_failure_data_analysis_as_success():
    msg_ok = ToolMessage(content="result", tool_call_id="a", name="data_analysis")
    msg_fail = ToolMessage(
        content=_make_failure_content("[PERMANENT]", "x"),
        tool_call_id="b",
        name="data_analysis",
    )
    msg_other = ToolMessage(content="ok", tool_call_id="c", name="text2sql")
    assert _tool_succeeded(msg_ok) is True
    assert _tool_succeeded(msg_fail) is False
    assert _tool_succeeded(msg_other) is True


def test_make_data_analysis_failure_response_async_uses_ainvoke():
    """In async mode the failure-response node must use ainvoke, not invoke."""
    import asyncio
    from unittest.mock import AsyncMock

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Sorry, analysis failed."))

    state = AgentState(
        messages=[
            HumanMessage(content="do analysis"),
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=_format_data_analysis_error(RuntimeError("boom")),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )

    node = make_data_analysis_failure_response(mock_llm, sync_mode=False)
    result = asyncio.run(node(state))

    assert "data_analysis_retries" not in result
    assert result["messages"][0].content == "Sorry, analysis failed."
    mock_llm.ainvoke.assert_called_once()
    assert not mock_llm.invoke.called
