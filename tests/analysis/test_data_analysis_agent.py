from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from openchatbi.analysis import agent as analysis_agent
from openchatbi.analysis.agent import (
    DATA_ANALYSIS_ERROR_PREFIX,
    DATA_ANALYSIS_FAILED_MARKER,
    DataAnalysisInput,
    _build_sub_agent_config,
    _classify_data_analysis_error,
    _extract_final_content,
    _format_data_analysis_error,
    _is_transient_error,
    _load_data_analysis_prompt,
    build_data_analysis_agent,
    get_data_analysis_tool,
)


@pytest.fixture
def mock_sql_graph():
    return MagicMock(spec=CompiledStateGraph)


def test_load_data_analysis_prompt_injects_min_input_length(monkeypatch):
    # The [min_input_length] placeholder is replaced with the resolved model minimum.
    monkeypatch.setattr(analysis_agent, "_resolve_min_input_length", lambda: 48)
    prompt = _load_data_analysis_prompt()
    assert "[min_input_length]" not in prompt
    assert "48 input points" in prompt


@patch("openchatbi.analysis.agent.create_deep_agent")
@patch("openchatbi.analysis.agent.get_analysis_llm")
@patch("openchatbi.agent_graph.get_sql_tools")
@patch("openchatbi.analysis.agent.check_forecast_service_health")
def test_build_data_analysis_agent(
    mock_check_health, mock_get_sql_tools, mock_get_llm, mock_create_deep_agent, mock_sql_graph
):
    # Setup mocks
    mock_check_health.return_value = True
    mock_text2sql = MagicMock(spec=StructuredTool)
    mock_text2sql.name = "text2sql"
    mock_get_sql_tools.return_value = mock_text2sql

    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    mock_agent = MagicMock(spec=CompiledStateGraph)
    mock_create_deep_agent.return_value = mock_agent

    # Call
    agent = build_data_analysis_agent(sql_graph=mock_sql_graph)

    # Assert
    assert agent == mock_agent
    mock_get_sql_tools.assert_called_once_with(sql_graph=mock_sql_graph, sync_mode=False)
    mock_get_llm.assert_called_once()

    # Check that create_deep_agent was called with correct arguments
    call_args = mock_create_deep_agent.call_args[1]
    assert call_args["model"] == mock_llm
    assert "system_prompt" in call_args

    # Check tools list
    tools = call_args["tools"]
    tool_names = [t.name if hasattr(t, "name") else t.__name__ for t in tools]
    assert "text2sql" in tool_names
    assert "run_python_code" in tool_names
    assert "timeseries_forecast" in tool_names
    assert "anomaly_detection" in tool_names
    assert "adtributor_drilldown" in tool_names


@patch("openchatbi.analysis.agent.build_data_analysis_agent")
def test_get_data_analysis_tool_sync(mock_build_agent, mock_sql_graph):
    # Setup mock agent
    mock_agent = MagicMock()
    # The agent returns a dict with "messages" key containing AIMessage
    mock_message = MagicMock()
    mock_message.content = "Analysis result"
    mock_agent.invoke.return_value = {"messages": [mock_message]}
    mock_build_agent.return_value = mock_agent

    # Call
    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)

    # Assert
    assert isinstance(tool, StructuredTool)
    assert tool.name == "data_analysis"
    assert tool.args_schema == DataAnalysisInput

    # Test invocation
    result = tool.invoke(
        {"reasoning": "test", "task": "analyze this"},
        config={"configurable": {"thread_id": "parent-123"}},
    )
    assert result == "Analysis result"
    mock_agent.invoke.assert_called_once()

    # The sub-agent must receive an isolated, derived thread_id (not the parent's).
    _, call_kwargs = mock_agent.invoke.call_args
    sub_config = call_kwargs["config"]
    assert sub_config["configurable"]["thread_id"] == "parent-123:data_analysis"


@pytest.mark.asyncio
@patch("openchatbi.analysis.agent.build_data_analysis_agent")
async def test_get_data_analysis_tool_async(mock_build_agent, mock_sql_graph):
    # Setup mock agent
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Async analysis result"

    # Create a coroutine that returns the dict
    async def mock_ainvoke(*args, **kwargs):
        return {"messages": [mock_message]}

    mock_agent.ainvoke = mock_ainvoke
    mock_build_agent.return_value = mock_agent

    # Call
    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=False)

    # Assert
    assert isinstance(tool, StructuredTool)
    assert tool.name == "data_analysis"

    # Test invocation
    result = await tool.ainvoke({"reasoning": "test", "task": "analyze this"})
    assert result == "Async analysis result"


@patch("openchatbi.analysis.agent.build_data_analysis_agent")
def test_sync_tool_propagates_graph_interrupt(mock_build_agent, mock_sql_graph):
    """GraphInterrupt from the sub-agent must bubble up (HITL), not be swallowed."""
    from langgraph.errors import GraphInterrupt

    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = GraphInterrupt(())
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)
    with pytest.raises(GraphInterrupt):
        tool.invoke({"reasoning": "test", "task": "analyze this"})


@pytest.mark.asyncio
@patch("openchatbi.analysis.agent.build_data_analysis_agent")
async def test_async_tool_propagates_graph_interrupt(mock_build_agent, mock_sql_graph):
    """GraphInterrupt from the async sub-agent must bubble up (HITL), not be swallowed."""
    from langgraph.errors import GraphInterrupt

    mock_agent = MagicMock()

    async def mock_ainvoke(*args, **kwargs):
        raise GraphInterrupt(())

    mock_agent.ainvoke = mock_ainvoke
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=False)
    with pytest.raises(GraphInterrupt):
        await tool.ainvoke({"reasoning": "test", "task": "analyze this"})


@patch("openchatbi.analysis.agent.build_data_analysis_agent")
def test_sync_tool_returns_structured_error_string_on_generic_exception(mock_build_agent, mock_sql_graph):
    """Non-interrupt exceptions are caught and returned as a structured, parent-parsable error string."""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("boom")
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)
    result = tool.invoke({"reasoning": "test", "task": "analyze this"})
    assert result.startswith(DATA_ANALYSIS_FAILED_MARKER)
    assert "[PERMANENT]" in result
    assert DATA_ANALYSIS_ERROR_PREFIX in result
    assert "boom" in result


def test_build_sub_agent_config_derives_isolated_thread_id():
    """A derived child thread_id is used; checkpoint_id is dropped and the
    checkpoint namespace is pinned to a fixed single-level value."""
    parent = {
        "configurable": {
            "thread_id": "abc",
            "checkpoint_ns": "use_tool:1",
            "checkpoint_id": "ckpt-1",
        },
        "tags": ["x"],
    }
    sub = _build_sub_agent_config(parent)
    assert sub["configurable"]["thread_id"] == "abc:data_analysis"
    # checkpoint_ns is pinned (not popped) so the sub-agent streams as a nested
    # subgraph; the parent's task-scoped namespace must not leak through.
    assert sub["configurable"]["checkpoint_ns"] == "data_analysis"
    # checkpoint_id reset is the real isolation guarantee.
    assert "checkpoint_id" not in sub["configurable"]
    # Non-configurable keys are propagated.
    assert sub["tags"] == ["x"]


def test_build_sub_agent_config_handles_none():
    """No parent config should still yield a valid (empty) configurable."""
    sub = _build_sub_agent_config(None)
    assert sub["configurable"] == {}


def test_build_sub_agent_config_drops_recursion_limit_and_metadata():
    """Parent/tool-node ``recursion_limit`` and ``metadata`` must not leak into
    the sub-agent; an explicit OpenChatBI recursion limit is applied instead."""
    from openchatbi.constants import DEFAULT_ANALYSIS_RECURSION_LIMIT

    parent = {
        "configurable": {"thread_id": "abc"},
        "recursion_limit": 25,
        "metadata": {"langfuse_trace_name": "parent"},
        "tags": ["x"],
    }
    sub = _build_sub_agent_config(parent)

    # The dangerous parent defaults are dropped.
    assert "metadata" not in sub
    assert sub["recursion_limit"] != 25
    # callbacks/tags still flow through.
    assert sub["tags"] == ["x"]
    # Defaults to the OpenChatBI package default when no custom config is loaded.
    assert sub["recursion_limit"] == DEFAULT_ANALYSIS_RECURSION_LIMIT


def test_build_sub_agent_config_uses_openchatbi_recursion_limit():
    """A custom ``analysis_recursion_limit`` in OpenChatBI config is honored."""
    from unittest.mock import MagicMock

    from openchatbi.config_loader import ConfigLoader

    ConfigLoader().set(
        {
            "default_llm": MagicMock(),
            "data_warehouse_config": {
                "uri": "sqlite:///:memory:",
                "include_tables": None,
                "database_name": "test_db",
            },
            "analysis_recursion_limit": 1234,
        }
    )

    sub = _build_sub_agent_config({"configurable": {"thread_id": "abc"}})
    assert sub["recursion_limit"] == 1234


def test_classify_data_analysis_error_marks_graph_recursion_as_retryable():
    """GraphRecursionError is retryable because simplifying the task often helps."""
    from langgraph.errors import GraphRecursionError

    tag, message = _classify_data_analysis_error(GraphRecursionError("limit hit"))
    assert tag == "[RETRYABLE]"
    assert "more steps than allowed" in message


def test_classify_data_analysis_error_marks_transient_errors_as_retryable():
    """Network/timeout/rate-limit style errors are retryable and preserve error details."""

    class FakeAPITimeoutError(Exception):
        pass

    tag, message = _classify_data_analysis_error(FakeAPITimeoutError("too slow"))
    assert tag == "[RETRYABLE]"
    assert "transient error" in message
    assert "too slow" in message


def test_is_transient_error_walks_exception_chain():
    """A transient error wrapped in a generic carrier is detected via __cause__."""

    class CarrierError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    carrier = CarrierError("api call failed")
    carrier.__cause__ = ReadTimeout("read timed out")
    assert _is_transient_error(carrier) is True


def test_is_transient_error_detects_retryable_status_codes():
    """HTTP 429 / 5xx status codes on the exception or error.response are treated as transient."""

    class HTTPStatusError(Exception):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.status_code = status_code

    class ResponseWrappedHTTPError(Exception):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.response = MagicMock(status_code=status_code)

    assert _is_transient_error(HTTPStatusError("rate", 429)) is True
    assert _is_transient_error(HTTPStatusError("server", 503)) is True
    assert _is_transient_error(HTTPStatusError("client", 404)) is False
    assert _is_transient_error(ResponseWrappedHTTPError("rate", 429)) is True
    assert _is_transient_error(ResponseWrappedHTTPError("server", 502)) is True
    assert _is_transient_error(ResponseWrappedHTTPError("client", 400)) is False


def test_is_transient_error_generic_returns_false():
    class TokenLimitError(Exception):
        pass

    assert _is_transient_error(ValueError("bad arg")) is False
    assert _is_transient_error(KeyError("missing")) is False
    assert _is_transient_error(TokenLimitError("context too long")) is False


def test_classify_data_analysis_error_marks_generic_errors_as_permanent():
    """Generic runtime errors are permanent (do not silently retry forever)."""
    tag, message = _classify_data_analysis_error(RuntimeError("boom"))
    assert tag == "[PERMANENT]"
    assert "boom" in message


def test_format_data_analysis_error_includes_marker_and_tag():
    """The formatted error can be parsed by the parent graph routing logic."""
    formatted = _format_data_analysis_error(RuntimeError("boom"))
    assert formatted.startswith(f"{DATA_ANALYSIS_FAILED_MARKER}[PERMANENT]")
    assert DATA_ANALYSIS_ERROR_PREFIX in formatted
    assert "boom" in formatted


def test_agent_prompt_includes_data_analysis_failure_policy():
    """The main agent prompt must instruct the model to surface data_analysis failures."""
    from openchatbi.prompts.system_prompt import get_agent_prompt_template, reset_cache

    reset_cache()
    prompt = get_agent_prompt_template()
    assert DATA_ANALYSIS_FAILED_MARKER in prompt
    assert "[PERMANENT]" in prompt
    assert "[RETRYABLE]" in prompt
    assert "Never silently redo the analysis" in prompt


def test_extract_final_content_string_and_blocks():
    str_msg = MagicMock()
    str_msg.content = "hello"
    assert _extract_final_content({"messages": [str_msg]}) == "hello"

    block_msg = MagicMock()
    block_msg.content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _extract_final_content({"messages": [block_msg]}) == "a\nb"

    assert _extract_final_content({"messages": []}) == "{'messages': []}"
