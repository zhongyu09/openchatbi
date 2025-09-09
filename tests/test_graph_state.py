"""Tests for graph state management."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openchatbi.graph_state import AgentState, InputState, OutputState


class TestAgentState:
    """Test AgentState functionality."""

    def test_agent_state_with_data(self):
        """Test creating AgentState with initial data."""
        messages = [HumanMessage(content="Test message")]
        sql = "SELECT * FROM test_table;"
        agent_next_node = "sql_generation"
        final_answer = "Here is your data"

        state = AgentState(messages=messages, sql=sql, agent_next_node=agent_next_node, final_answer=final_answer)

        assert state["messages"] == messages
        assert state["sql"] == sql
        assert state["agent_next_node"] == agent_next_node
        assert state["final_answer"] == final_answer

    def test_agent_state_message_types(self):
        """Test AgentState with different message types."""
        messages = [
            HumanMessage(content="User question"),
            AIMessage(content="AI response"),
            ToolMessage(content="Tool result", tool_call_id="test_id"),
        ]

        state = AgentState(messages=messages)

        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], ToolMessage)

    def test_agent_state_immutability(self):
        """Test that AgentState behaves correctly with updates."""
        original_state = AgentState(
            messages=[HumanMessage(content="Original")],
            sql="SELECT 1;",
            agent_next_node="original_node",
            final_answer="Original answer",
        )

        # Create updated state
        new_messages = original_state["messages"] + [AIMessage(content="Response")]
        updated_state = AgentState(
            messages=new_messages, sql="SELECT 2;", agent_next_node="updated_node", final_answer="Updated answer"
        )

        # Original state should remain unchanged
        assert len(original_state["messages"]) == 1
        assert original_state["sql"] == "SELECT 1;"
        assert original_state["agent_next_node"] == "original_node"
        assert original_state["final_answer"] == "Original answer"

        # Updated state should have new values
        assert len(updated_state["messages"]) == 2
        assert updated_state["sql"] == "SELECT 2;"
        assert updated_state["agent_next_node"] == "updated_node"
        assert updated_state["final_answer"] == "Updated answer"


class TestInputState:
    """Test InputState functionality."""

    def test_input_state_creation(self):
        """Test creating InputState."""
        messages = [HumanMessage(content="Input message")]

        state = InputState(messages=messages)

        assert state["messages"] == messages

    def test_input_state_empty_messages(self):
        """Test InputState with empty messages."""
        state = InputState(messages=[])

        assert state["messages"] == []


class TestOutputState:
    """Test OutputState functionality."""

    def test_output_state_creation(self):
        """Test creating OutputState."""
        messages = [AIMessage(content="Output message")]

        state = OutputState(messages=messages)

        assert state["messages"] == messages

    def test_output_state_with_multiple_messages(self):
        """Test OutputState with conversation history."""
        messages = [
            HumanMessage(content="Question"),
            AIMessage(content="Answer"),
            HumanMessage(content="Follow-up"),
            AIMessage(content="Final response"),
        ]

        state = OutputState(messages=messages)

        assert len(state["messages"]) == 4
        assert state["messages"] == messages


class TestStateIntegration:
    """Test integration between different state types."""

    def test_input_to_agent_state_conversion(self):
        """Test converting InputState to AgentState."""
        input_messages = [HumanMessage(content="User input")]
        input_state = InputState(messages=input_messages)

        # Simulate conversion to AgentState
        agent_state = AgentState(messages=input_state["messages"], sql="", agent_next_node="", final_answer="")

        assert agent_state["messages"] == input_messages
        assert agent_state["sql"] == ""

    def test_agent_to_output_state_conversion(self):
        """Test converting AgentState to OutputState."""
        agent_messages = [HumanMessage(content="Question"), AIMessage(content="Generated response")]

        agent_state = AgentState(
            messages=agent_messages,
            sql="SELECT * FROM test_table;",
            agent_next_node="output",
            final_answer="Generated response",
        )

        # Simulate conversion to OutputState
        output_state = OutputState(messages=agent_state["messages"])

        assert output_state["messages"] == agent_messages

    def test_state_serialization_compatibility(self):
        """Test that states can be serialized and deserialized."""
        original_state = AgentState(
            messages=[HumanMessage(content="Test"), AIMessage(content="Response")],
            sql="SELECT COUNT(*) FROM table1;",
            agent_next_node="final",
            final_answer="Count results",
        )

        # Convert to dict (simulating serialization)
        state_dict = {
            "messages": original_state["messages"],
            "sql": original_state["sql"],
            "agent_next_node": original_state["agent_next_node"],
            "final_answer": original_state["final_answer"],
        }

        # Recreate from dict (simulating deserialization)
        recreated_state = AgentState(**state_dict)

        assert recreated_state["messages"] == original_state["messages"]
        assert recreated_state["sql"] == original_state["sql"]
        assert recreated_state["agent_next_node"] == original_state["agent_next_node"]
        assert recreated_state["final_answer"] == original_state["final_answer"]
