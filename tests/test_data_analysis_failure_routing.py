"""Tests for data_analysis failure routing and surfacing in the main graph."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

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


def test_data_analysis_retry_hint_increments_counter_and_updates_tool_message_in_place():
    original_content = _make_failure_content("[RETRYABLE]", "too many steps")
    initial_messages = add_messages(
        [],
        [
            HumanMessage(content="analyze sales"),
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {}, "id": "t1"}]),
            ToolMessage(
                content=original_content,
                tool_call_id="t1",
                name="data_analysis",
                id="msg-t1",
            ),
        ],
    )
    state = AgentState(messages=initial_messages, data_analysis_retries=0)

    result = data_analysis_retry_hint(state)
    assert result["data_analysis_retries"] == 1
    assert len(result["messages"]) == 1

    updated = result["messages"][0]
    assert isinstance(updated, ToolMessage)
    assert updated.id == "msg-t1"
    assert updated.tool_call_id == "t1"
    assert updated.name == "data_analysis"
    assert updated.content.startswith(original_content)
    assert "The previous `data_analysis` attempt failed and is retryable." in updated.content
    assert "Failure: too many steps" in updated.content
    assert "You may call `data_analysis` ONE more time" in updated.content

    # Verify LangGraph's add_messages reducer replaces the ToolMessage in-place
    # instead of appending a new message or introducing a trailing SystemMessage.
    merged_messages = add_messages(initial_messages, result["messages"])
    assert len(merged_messages) == 3
    assert isinstance(merged_messages[-1], ToolMessage)
    assert merged_messages[-1].id == "msg-t1"
    assert merged_messages[-1].content == updated.content
    assert not any(isinstance(m, SystemMessage) for m in merged_messages)
    # The updated ToolMessage must still be parseable as a [RETRYABLE] failure.
    parsed = _parse_data_analysis_failure(merged_messages[-1].content)
    assert parsed is not None
    assert parsed[0] == "[RETRYABLE]"


def test_data_analysis_retry_hint_compatible_with_anthropic_message_formatting():
    """In-place ToolMessage update must not break Anthropic's strict system-message ordering."""
    from langchain_anthropic.chat_models import _format_messages

    initial_messages = add_messages(
        [],
        [
            HumanMessage(content="analyze sales"),
            AIMessage(content="", tool_calls=[{"name": "data_analysis", "args": {"task": "x"}, "id": "t1"}]),
            ToolMessage(
                content=_make_failure_content("[RETRYABLE]", "recursion limit"),
                tool_call_id="t1",
                name="data_analysis",
            ),
        ],
    )
    state = AgentState(messages=initial_messages, data_analysis_retries=0)
    result = data_analysis_retry_hint(state)
    merged_messages = add_messages(initial_messages, result["messages"])

    system_prompt, formatted = _format_messages([SystemMessage(content="Top system prompt")] + merged_messages)
    assert system_prompt == "Top system prompt"
    assert len(formatted) == 3
    assert formatted[-1]["role"] == "user"


def test_data_analysis_retry_hint_when_no_failure_present_only_increments_counter():
    """Fallback branch when _latest_data_analysis_failure returns None."""
    state = AgentState(
        messages=[HumanMessage(content="analyze sales")],
        data_analysis_retries=1,
    )
    result = data_analysis_retry_hint(state)
    assert result == {"data_analysis_retries": 2}


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
    assert result["final_answer"] == "Sorry, analysis failed."
    assert result["history_messages"] == [AIMessage(content="Sorry, analysis failed.")]
    # The failure-response node should invoke the raw LLM with streaming_tokens=True, not a tool-bound one.
    mock_llm.invoke.assert_called_once()
    _, kwargs = mock_llm.invoke.call_args
    assert kwargs.get("config") == {"metadata": {"streaming_tokens": True}}
    assert not mock_llm.bind_tools.called


def test_reset_turn_state_resets_retry_counter():
    """reset_turn_state zeroes the retry budget for each new user invocation."""
    state = AgentState(messages=[HumanMessage(content="hi")], data_analysis_retries=3)
    result = reset_turn_state(state)
    assert result == {"data_analysis_retries": 0}


def test_route_after_tool_parallel_success_goes_to_llm_node():
    """When a sibling tool succeeded in the same batch as a permanent data_analysis
    failure, route back to llm_node so the successful result is not dropped."""
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


def test_route_after_tool_parallel_retryable_consumes_retry_budget_via_retry_hint():
    """When data_analysis fails with [RETRYABLE] alongside a successful sibling tool,
    route through data_analysis_retry_hint (which transitions to llm_node) so the
    retry budget is still decremented while preserving sibling tool results."""
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
                content=_make_failure_content("[RETRYABLE]", "timeout"),
                tool_call_id="t2",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "data_analysis_retry_hint"

    # Once retries are exhausted, a parallel batch with a successful sibling routes to llm_node.
    exhausted_state = AgentState(
        messages=state["messages"],
        data_analysis_retries=1,
    )
    assert route_after_tool(exhausted_state) == "llm_node"


def test_route_after_tool_parallel_failed_sibling_goes_to_failure_response():
    """If the sibling tool also failed, do not treat it as a successful sibling."""
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "text2sql", "args": {}, "id": "t1"},
                    {"name": "data_analysis", "args": {}, "id": "t2"},
                ],
            ),
            ToolMessage(
                content="Error occurred when calling Text2SQL tool.",
                tool_call_id="t1",
                name="text2sql",
            ),
            ToolMessage(
                content=_make_failure_content("[PERMANENT]", "boom"),
                tool_call_id="t2",
                name="data_analysis",
            ),
        ],
        data_analysis_retries=0,
    )
    assert route_after_tool(state) == "data_analysis_failure_response"


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
    msg_status_err = ToolMessage(content="err", tool_call_id="d", name="search_schema", status="error")
    msg_sql_err = ToolMessage(
        content="Error occurred when calling Text2SQL tool.",
        tool_call_id="e",
        name="text2sql",
    )
    assert _tool_succeeded(msg_ok) is True
    assert _tool_succeeded(msg_fail) is False
    assert _tool_succeeded(msg_other) is True
    assert _tool_succeeded(msg_status_err) is False
    assert _tool_succeeded(msg_sql_err) is False


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
    assert result["final_answer"] == "Sorry, analysis failed."
    assert result["history_messages"] == [AIMessage(content="Sorry, analysis failed.")]
    mock_llm.ainvoke.assert_called_once()
    _, kwargs = mock_llm.ainvoke.call_args
    assert kwargs.get("config") == {"metadata": {"streaming_tokens": True}}
    assert not mock_llm.invoke.called
