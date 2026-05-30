import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from openchatbi.agent_graph import build_agent_graph_sync, build_agent_graph_async
from openchatbi.catalog import CatalogStore

@pytest.fixture
def mock_catalog():
    return MagicMock(spec=CatalogStore)

    @patch("openchatbi.agent_graph.get_data_analysis_tool")
    @patch("openchatbi.agent_graph.get_sql_tools")
    @patch("openchatbi.agent_graph.build_sql_graph")
    @patch("openchatbi.agent_graph.create_mcp_tools_sync")
    def test_build_agent_graph_sync_includes_data_analysis(
        mock_create_mcp_tools_sync,
        mock_build_sql_graph,
        mock_get_sql_tools,
        mock_get_data_analysis_tool,
        mock_catalog
    ):
        # Setup mocks
        mock_create_mcp_tools_sync.return_value = []
    mock_sql_graph = MagicMock(spec=CompiledStateGraph)
    mock_build_sql_graph.return_value = mock_sql_graph
    mock_get_sql_tools.return_value = MagicMock()
    
    mock_data_analysis_tool = MagicMock()
    mock_data_analysis_tool.name = "data_analysis"
    mock_get_data_analysis_tool.return_value = mock_data_analysis_tool

    # Call
    graph = build_agent_graph_sync(catalog=mock_catalog, enable_context_management=False)

    # Assert
    assert isinstance(graph, CompiledStateGraph)
    mock_get_data_analysis_tool.assert_called_once()
    
    # We can't easily inspect the internal tools list of the compiled graph without
    # digging into its internal structure, but we verified the tool factory was called.

    @pytest.mark.asyncio
    @patch("openchatbi.agent_graph.get_data_analysis_tool")
    @patch("openchatbi.agent_graph.get_sql_tools")
    @patch("openchatbi.agent_graph.build_sql_graph")
    @patch("openchatbi.agent_graph.get_mcp_tools_async")
    async def test_build_agent_graph_async_includes_data_analysis(
        mock_get_mcp_tools_async,
        mock_build_sql_graph,
        mock_get_sql_tools,
        mock_get_data_analysis_tool,
        mock_catalog
    ):
        # Setup mocks
        mock_get_mcp_tools_async.return_value = []
        mock_sql_graph = MagicMock(spec=CompiledStateGraph)
        mock_build_sql_graph.return_value = mock_sql_graph
        mock_get_sql_tools.return_value = MagicMock()
        
        mock_data_analysis_tool = MagicMock()
        mock_data_analysis_tool.name = "data_analysis"
        mock_get_data_analysis_tool.return_value = mock_data_analysis_tool

        # Call
        graph = await build_agent_graph_async(catalog=mock_catalog, enable_context_management=False)

        # Assert
        assert isinstance(graph, CompiledStateGraph)
        mock_get_data_analysis_tool.assert_called_once()
