"""Tests for incomplete tool call recovery functionality."""

from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openchatbi.agent_graph import agent_llm_call
from openchatbi.utils import recover_incomplete_tool_calls
from openchatbi.graph_state import AgentState


class TestIncompleteToolCallRecovery:
    """Test cases for recover_incomplete_tool_calls function."""

    def test_no_messages(self):
        """Test recovery with empty message list."""
        state = AgentState(messages=[])
        result = recover_incomplete_tool_calls(state)
        assert result == []

    def test_no_tool_calls(self):
        """Test recovery with messages but no tool calls."""
        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)
        assert result == []

    def test_complete_tool_calls(self):
        """Test recovery when all tool calls have responses."""
        messages = [
            HumanMessage(content="Search for data"),
            AIMessage(
                content="I'll search for that data.",
                tool_calls=[{"name": "search", "args": {"query": "data"}, "id": "call_1"}],
            ),
            ToolMessage(content="Search completed", tool_call_id="call_1"),
        ]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)
        assert result == []

    def test_incomplete_single_tool_call(self):
        """Test recovery when there's one incomplete tool call."""
        messages = [
            HumanMessage(content="Search for data"),
            AIMessage(
                content="I'll search for that data.",
                tool_calls=[{"name": "search", "args": {"query": "data"}, "id": "call_1"}],
            ),
        ]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)

        assert isinstance(result, list)
        assert len(result) == 1  # Just the recovery message

        failure_msg = result[0]
        assert failure_msg.tool_call_id == "call_1"
        assert "interrupted" in failure_msg.content.lower()

    def test_incomplete_multiple_tool_calls(self):
        """Test recovery when there are multiple incomplete tool calls."""
        messages = [
            HumanMessage(content="Search and analyze"),
            AIMessage(
                content="I'll search and analyze.",
                tool_calls=[
                    {"name": "search", "args": {"query": "data"}, "id": "call_1"},
                    {"name": "analyze", "args": {"data": "result"}, "id": "call_2"},
                ],
            ),
        ]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)

        assert isinstance(result, list)
        assert len(result) == 2  # Just the recovery messages

        # Check that both tool calls get failure messages
        recovery_messages = result
        tool_call_ids = {msg.tool_call_id for msg in recovery_messages}
        assert tool_call_ids == {"call_1", "call_2"}

        for msg in recovery_messages:
            assert isinstance(msg, ToolMessage)
            assert "interrupted" in msg.content.lower()

    def test_partial_incomplete_tool_calls(self):
        """Test recovery when some tool calls are complete, others are not."""
        messages = [
            HumanMessage(content="Search and analyze"),
            AIMessage(
                content="I'll search and analyze.",
                tool_calls=[
                    {"name": "search", "args": {"query": "data"}, "id": "call_1"},
                    {"name": "analyze", "args": {"data": "result"}, "id": "call_2"},
                ],
            ),
            ToolMessage(content="Search completed", tool_call_id="call_1"),
            # Missing ToolMessage for call_2
        ]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)

        assert isinstance(result, list)
        assert len(result) == 3  # RemoveMessage + recovery message + re-added message

        # Should have: RemoveMessage, ToolMessage(recovery for call_2), ToolMessage(original for call_1)
        operations = result
        assert "RemoveMessage" in str(type(operations[0]))  # Remove the existing ToolMessage
        assert isinstance(operations[1], ToolMessage)  # Recovery message for call_2
        assert isinstance(operations[2], ToolMessage)  # Re-added original message for call_1

        # The recovery message should be for call_2
        recovery_msg = operations[1]
        assert recovery_msg.tool_call_id == "call_2"
        assert "interrupted" in recovery_msg.content.lower()

        # The re-added message should be the original for call_1
        original_msg = operations[2]
        assert original_msg.tool_call_id == "call_1"
        assert original_msg.content == "Search completed"

    def test_multiple_ai_messages_with_tool_calls(self):
        """Test recovery considers only the last AIMessage with tool calls."""
        messages = [
            HumanMessage(content="First task"),
            AIMessage(content="Doing first task.", tool_calls=[{"name": "task1", "args": {}, "id": "old_call"}]),
            ToolMessage(content="Task 1 done", tool_call_id="old_call"),
            HumanMessage(content="Second task"),
            AIMessage(content="Doing second task.", tool_calls=[{"name": "task2", "args": {}, "id": "new_call"}]),
            # Missing ToolMessage for new_call
        ]
        state = AgentState(messages=messages)
        result = recover_incomplete_tool_calls(state)

        assert isinstance(result, list)
        assert len(result) == 1  # Just the recovery message

        # The recovery message should be for new_call only
        recovery_msg = result[0]
        assert recovery_msg.tool_call_id == "new_call"
        assert "interrupted" in recovery_msg.content.lower()

    def test_llm_node_integration_with_recovery(self):
        """Test that the llm_node handles recovery correctly and continues processing."""
        # Create a mock llm_node function for testing
        mock_llm = Mock()
        mock_tools = []
        llm_node_func = agent_llm_call(mock_llm, mock_tools)

        # State with incomplete tool calls
        messages = [
            HumanMessage(content="Search for data"),
            AIMessage(
                content="I'll search for that data.",
                tool_calls=[{"name": "search", "args": {"query": "data"}, "id": "call_1"}],
            ),
        ]
        state = AgentState(messages=messages)

        # Call the llm node - it should detect incomplete tool calls and return recovery
        result = llm_node_func(state)

        # Should return message operations and continue to llm_node
        assert "messages" in result
        assert "agent_next_node" in result
        assert result["agent_next_node"] == "llm_node"

        # Should have recovery ToolMessage operation for the incomplete call
        operations = result["messages"]
        assert len(operations) == 1  # Only recovery message needed
        assert isinstance(operations[0], ToolMessage)
        assert operations[0].tool_call_id == "call_1"
