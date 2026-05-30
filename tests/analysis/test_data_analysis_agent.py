import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from openchatbi.analysis.agent import (
    build_data_analysis_agent,
    get_data_analysis_tool,
    DataAnalysisInput,
)

@pytest.fixture
def mock_sql_graph():
    return MagicMock(spec=CompiledStateGraph)

@patch("openchatbi.analysis.agent.create_deep_agent")
@patch("openchatbi.analysis.agent.get_llm")
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
    # Deep agent returns a dict with "messages" key containing AIMessage
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
    result = tool.invoke({"reasoning": "test", "task": "analyze this"})
    assert result == "Analysis result"
    mock_agent.invoke.assert_called_once()

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
        mock_agent.ainvoke.assert_called_once()
