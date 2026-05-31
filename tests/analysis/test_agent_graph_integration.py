import pytest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from openchatbi.agent_graph import build_agent_graph_sync, build_agent_graph_async
from openchatbi.catalog import CatalogStore


def _make_tool(name: str) -> StructuredTool:
    def mock_func(input_str: str) -> str:
        return f"Mock {name} result"

    return StructuredTool.from_function(func=mock_func, name=name, description=f"Mock {name} tool")


@pytest.fixture
def mock_catalog():
    return MagicMock(spec=CatalogStore)


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.bind_tools = Mock(return_value=llm)
    return llm


@patch("openchatbi.agent_graph.get_memory_tools", return_value=[])
@patch("openchatbi.analysis.agent.get_data_analysis_tool")
@patch("openchatbi.agent_graph.get_sql_tools")
@patch("openchatbi.agent_graph.build_sql_graph")
@patch("openchatbi.agent_graph.create_mcp_tools_sync")
def test_build_agent_graph_sync_includes_data_analysis(
    mock_create_mcp_tools_sync,
    mock_build_sql_graph,
    mock_get_sql_tools,
    mock_get_data_analysis_tool,
    mock_get_memory_tools,
    mock_catalog,
    mock_llm,
):
    # Setup mocks
    mock_create_mcp_tools_sync.return_value = []
    mock_sql_graph = MagicMock(spec=CompiledStateGraph)
    mock_build_sql_graph.return_value = mock_sql_graph
    mock_get_sql_tools.return_value = _make_tool("text2sql")
    mock_get_data_analysis_tool.return_value = _make_tool("data_analysis")

    # Call
    with patch("openchatbi.agent_graph.get_llm", return_value=mock_llm):
        graph = build_agent_graph_sync(catalog=mock_catalog, enable_context_management=False)

    # Assert
    assert isinstance(graph, CompiledStateGraph)
    mock_get_data_analysis_tool.assert_called_once()


@pytest.mark.asyncio
@patch("openchatbi.agent_graph.get_memory_tools", return_value=[])
@patch("openchatbi.analysis.agent.get_data_analysis_tool")
@patch("openchatbi.agent_graph.get_sql_tools")
@patch("openchatbi.agent_graph.build_sql_graph")
@patch("openchatbi.agent_graph.get_mcp_tools_async")
async def test_build_agent_graph_async_includes_data_analysis(
    mock_get_mcp_tools_async,
    mock_build_sql_graph,
    mock_get_sql_tools,
    mock_get_data_analysis_tool,
    mock_get_memory_tools,
    mock_catalog,
    mock_llm,
):
    # Setup mocks
    mock_get_mcp_tools_async.return_value = []
    mock_sql_graph = MagicMock(spec=CompiledStateGraph)
    mock_build_sql_graph.return_value = mock_sql_graph
    mock_get_sql_tools.return_value = _make_tool("text2sql")
    mock_get_data_analysis_tool.return_value = _make_tool("data_analysis")

    # Call
    with patch("openchatbi.agent_graph.get_llm", return_value=mock_llm):
        graph = await build_agent_graph_async(catalog=mock_catalog, enable_context_management=False)

    # Assert
    assert isinstance(graph, CompiledStateGraph)
    mock_get_data_analysis_tool.assert_called_once()
