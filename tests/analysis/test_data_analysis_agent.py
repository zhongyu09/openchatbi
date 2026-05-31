from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from openchatbi.analysis.agent import (
    DataAnalysisInput,
    _build_sub_agent_config,
    _extract_final_content,
    build_data_analysis_agent,
    get_data_analysis_tool,
)


@pytest.fixture
def mock_sql_graph():
    return MagicMock(spec=CompiledStateGraph)


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
def test_sync_tool_returns_error_string_on_generic_exception(mock_build_agent, mock_sql_graph):
    """Non-interrupt exceptions are caught and returned as an error string."""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("boom")
    mock_build_agent.return_value = mock_agent

    tool = get_data_analysis_tool(sql_graph=mock_sql_graph, sync_mode=True)
    result = tool.invoke({"reasoning": "test", "task": "analyze this"})
    assert "Error occurred during data analysis" in result
    assert "boom" in result


def test_build_sub_agent_config_derives_isolated_thread_id():
    """A derived child thread_id should be used, and stale checkpoint keys dropped."""
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
    assert "checkpoint_ns" not in sub["configurable"]
    assert "checkpoint_id" not in sub["configurable"]
    # Non-configurable keys are propagated.
    assert sub["tags"] == ["x"]


def test_build_sub_agent_config_handles_none():
    """No parent config should still yield a valid (empty) configurable."""
    sub = _build_sub_agent_config(None)
    assert sub["configurable"] == {}


def test_extract_final_content_string_and_blocks():
    str_msg = MagicMock()
    str_msg.content = "hello"
    assert _extract_final_content({"messages": [str_msg]}) == "hello"

    block_msg = MagicMock()
    block_msg.content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _extract_final_content({"messages": [block_msg]}) == "a\nb"

    assert _extract_final_content({"messages": []}) == "{'messages': []}"
