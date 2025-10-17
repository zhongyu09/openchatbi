"""Integration tests for agent graph with context management."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from openchatbi.agent_graph import _build_graph_core, agent_llm_call, build_agent_graph_sync, build_agent_graph_async
from openchatbi.context_manager import ContextManager
from openchatbi.context_config import ContextConfig

from openchatbi.graph_state import AgentState


class TestAgentGraphIntegration:
    """Integration tests for agent graph with context management."""

    @pytest.fixture
    def mock_catalog(self):
        """Mock catalog store for testing."""
        catalog = Mock()
        catalog.get_schema = Mock(return_value={"tables": []})
        return catalog

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        llm.bind_tools = Mock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_tools(self):
        """Mock tools for testing."""

        def mock_tool_func(query: str) -> str:
            return "Mock tool result"

        tool = StructuredTool.from_function(func=mock_tool_func, name="mock_tool", description="Mock tool for testing")
        return [tool]

    @pytest.fixture
    def test_config(self):
        """Test configuration for context management."""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=800,
            keep_recent_messages=3,
            max_tool_output_length=100,
        )

    def test_agent_llm_node_with_context_manager(self, mock_llm, mock_tools, test_config):
        """Test agent llm_node with context manager integration."""
        context_manager = ContextManager(llm=mock_llm, config=test_config)

        # Mock LLM response
        mock_response = AIMessage(content="Test response", tool_calls=[])
        with patch("openchatbi.agent_graph.call_llm_chat_model_with_retry", return_value=mock_response):
            llm_node_func = agent_llm_call(mock_llm, mock_tools, context_manager)

            # Create test state with long messages to trigger context management
            long_messages = [
                HumanMessage(content="A" * 500),  # Long message
                AIMessage(content="B" * 500),  # Long message
                ToolMessage(content="C" * 200, tool_call_id="123"),  # Long tool output
                HumanMessage(content="Recent question"),
            ]

            state = AgentState(messages=long_messages)
            result = llm_node_func(state)

            # Should have processed the state
            assert "messages" in result
            assert isinstance(result["messages"][0], AIMessage)

    def test_agent_llm_node_without_context_manager(self, mock_llm, mock_tools):
        """Test agent llm_node without context manager."""
        mock_response = AIMessage(content="Test response", tool_calls=[])
        with patch("openchatbi.agent_graph.call_llm_chat_model_with_retry", return_value=mock_response):
            llm_node_func = agent_llm_call(mock_llm, mock_tools, context_manager=None)

            state = AgentState(messages=[HumanMessage(content="Test")])
            result = llm_node_func(state)

            assert "messages" in result
            assert isinstance(result["messages"][0], AIMessage)

    def test_build_graph_core_with_context_management(self, mock_catalog, mock_llm):
        """Test core graph building with context management enabled."""

        def create_mock_tool(name):
            def mock_func(input_str: str) -> str:
                return f"Mock {name} result"

            return StructuredTool.from_function(func=mock_func, name=name, description=f"Mock {name} tool")

        # Mock all the tool imports directly
        with (
            patch("openchatbi.agent_graph.search_knowledge", create_mock_tool("search_knowledge")),
            patch("openchatbi.agent_graph.show_schema", create_mock_tool("show_schema")),
            patch("openchatbi.agent_graph.run_python_code", create_mock_tool("run_python_code")),
            patch("openchatbi.agent_graph.save_report", create_mock_tool("save_report")),
            patch("openchatbi.agent_graph.get_sql_tools") as mock_get_sql_tools,
            patch("openchatbi.agent_graph.build_sql_graph") as mock_sql_graph,
            patch("openchatbi.agent_graph.get_memory_tools") as mock_memory_tools,
            patch("openchatbi.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("openchatbi.agent_graph.get_default_llm", return_value=mock_llm),
        ):

            # Setup function-based mocks
            mock_get_sql_tools.return_value = create_mock_tool("call_sql_graph_tool")
            mock_sql_graph.return_value = Mock()
            mock_memory_tools.return_value = (
                create_mock_tool("manage_memory_tool"),
                create_mock_tool("search_memory_tool"),
            )
            mock_mcp_tools.return_value = []

            graph = _build_graph_core(
                catalog=mock_catalog,
                sync_mode=True,
                checkpointer=None,
                memory_store=None,
                memory_tools=None,
                mcp_tools=[],
                enable_context_management=True,
            )

            # Should create a compiled graph
            assert graph is not None
            # Verify that SQL graph was initialized
            mock_sql_graph.assert_called_once()

    def test_build_graph_core_without_context_management(self, mock_catalog, mock_llm):
        """Test core graph building with context management disabled."""

        def create_mock_tool(name):
            def mock_func(input_str: str) -> str:
                return f"Mock {name} result"

            return StructuredTool.from_function(func=mock_func, name=name, description=f"Mock {name} tool")

        # Mock all the tool imports directly - same pattern as with context management
        with (
            patch("openchatbi.agent_graph.search_knowledge", create_mock_tool("search_knowledge")),
            patch("openchatbi.agent_graph.show_schema", create_mock_tool("show_schema")),
            patch("openchatbi.agent_graph.run_python_code", create_mock_tool("run_python_code")),
            patch("openchatbi.agent_graph.save_report", create_mock_tool("save_report")),
            patch("openchatbi.agent_graph.get_sql_tools") as mock_get_sql_tools,
            patch("openchatbi.agent_graph.build_sql_graph") as mock_sql_graph,
            patch("openchatbi.agent_graph.get_memory_tools") as mock_memory_tools,
            patch("openchatbi.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("openchatbi.agent_graph.get_default_llm", return_value=mock_llm),
        ):

            # Setup function-based mocks
            mock_get_sql_tools.return_value = create_mock_tool("call_sql_graph_tool")
            mock_sql_graph.return_value = Mock()
            mock_memory_tools.return_value = (
                create_mock_tool("manage_memory_tool"),
                create_mock_tool("search_memory_tool"),
            )
            mock_mcp_tools.return_value = []

            graph = _build_graph_core(
                catalog=mock_catalog,
                sync_mode=True,
                checkpointer=None,
                memory_store=None,
                memory_tools=None,
                mcp_tools=[],
                enable_context_management=False,
            )

            # Should still create a compiled graph
            assert graph is not None

    def test_build_agent_graph_sync_with_context_management(self, mock_catalog):
        """Test sync graph building with context management."""
        with (
            patch("openchatbi.agent_graph.create_mcp_tools_sync") as mock_mcp_tools,
            patch("openchatbi.agent_graph._build_graph_core") as mock_build_core,
        ):

            mock_build_core.return_value = Mock()
            mock_mcp_tools.return_value = []

            graph = build_agent_graph_sync(catalog=mock_catalog, enable_context_management=True)

            # Verify _build_graph_core was called with correct parameters
            mock_build_core.assert_called_once()
            call_args = mock_build_core.call_args
            assert call_args[1]["enable_context_management"] is True

            # Should return the graph
            assert graph is not None

    @pytest.mark.asyncio
    async def test_build_agent_graph_async_with_context_management(self, mock_catalog):
        """Test async graph building with context management."""
        with (
            patch("openchatbi.agent_graph.get_mcp_tools_async") as mock_mcp_tools,
            patch("openchatbi.agent_graph._build_graph_core") as mock_build_core,
        ):

            mock_build_core.return_value = Mock()
            # Mock async function
            mock_mcp_tools.return_value = []

            graph = await build_agent_graph_async(catalog=mock_catalog, enable_context_management=True)

            # Verify _build_graph_core was called with correct parameters
            mock_build_core.assert_called_once()
            call_args = mock_build_core.call_args
            assert call_args[1]["enable_context_management"] is True

            # Should return the graph
            assert graph is not None

    @patch("openchatbi.agent_graph.call_llm_chat_model_with_retry")
    def test_full_context_management_flow(self, mock_llm_call, mock_catalog):
        """Test full context management flow in agent graph."""
        # Mock LLM responses
        mock_llm_call.side_effect = [
            AIMessage(content="Response 1"),
            AIMessage(content="Summary of conversation"),  # For summarization
            AIMessage(content="Final response"),
        ]

        context_manager = ContextManager(
            llm=Mock(),
            config=ContextConfig(
                enabled=True,
                summary_trigger_tokens=80,
                keep_recent_messages=2,
            ),
        )

        # Create many messages to trigger context management
        messages = []
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}" * 10),  # Make messages longer
                    AIMessage(content=f"Response {i}" * 10),
                    ToolMessage(content=f"Tool result {i}" * 20, tool_call_id=f"tool_{i}"),
                ]
            )

        # Test context management
        original_count = len(messages)
        context_manager.manage_context_messages(messages)
        managed_messages = messages

        # Should have fewer messages than input
        assert len(managed_messages) < original_count

        # Should preserve recent messages
        assert any("Question 9" in str(msg.content) for msg in managed_messages if hasattr(msg, "content"))


class TestContextManagementEdgeCases:
    """Test edge cases for context management in agent graph."""

    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        messages = []
        context_manager.manage_context_messages(messages)
        result = messages
        assert result == []

    def test_state_message_type_validation(self):
        """Test that only valid state message types are maintained during context management."""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        # State should only contain valid message types (no SystemMessage)
        messages = [
            HumanMessage(content="A" * 100),  # Long message
            AIMessage(content="B" * 100),  # Long message
            HumanMessage(content="Recent question"),
        ]

        with patch(
            "openchatbi.context_manager.call_llm_chat_model_with_retry", return_value=AIMessage(content="Summary")
        ):
            context_manager.manage_context_messages(messages)
            result = messages

        # Should only contain valid state message types
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(type(msg) in valid_types for msg in result), "Should only contain valid state message types"

    def test_context_management_with_tool_calls(self):
        """Test context management when AI messages have tool calls."""
        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        ai_message_with_tools = AIMessage(
            content="I'll help you with that.",
            tool_calls=[{"name": "search_tool", "args": {"query": "test"}, "id": "call_123"}],
        )

        messages = [ai_message_with_tools, HumanMessage(content="Follow up")]

        context_manager.manage_context_messages(messages)
        result = messages

        # AI message with tool calls should be preserved
        ai_msgs = [msg for msg in result if isinstance(msg, AIMessage)]
        assert len(ai_msgs) > 0
        assert any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in ai_msgs)

    @patch("openchatbi.context_manager.call_llm_chat_model_with_retry")
    def test_summarization_failure_fallback(self, mock_llm_call):
        """Test fallback behavior when summarization fails."""
        # Mock LLM failure
        mock_llm_call.side_effect = Exception("LLM unavailable")

        config = ContextConfig(enabled=True)
        context_manager = ContextManager(llm=Mock(), config=config)

        # Create messages that would trigger summarization (no SystemMessage in state)
        messages = [
            HumanMessage(content="A" * 100),  # Long messages to trigger
            AIMessage(content="B" * 100),
            HumanMessage(content="C" * 100),
            AIMessage(content="D" * 100),
            HumanMessage(content="Recent"),
        ]

        context_manager.manage_context_messages(messages)
        result = messages

        # Should fallback to sliding window
        assert len(result) <= len(messages)
        # Should preserve recent messages and only contain valid state message types
        assert any("Recent" in str(msg.content) for msg in result if hasattr(msg, "content"))
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(type(msg) in valid_types for msg in result), "Should only contain valid state message types"
