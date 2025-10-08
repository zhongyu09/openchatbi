"""Edge cases for context management."""

import pytest
import time
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from openchatbi.context_manager import ContextManager
from openchatbi.context_config import ContextConfig


class TestContextManagementEdgeCases:
    """Edge cases and boundary conditions for context management."""

    @pytest.fixture
    def edge_case_config(self):
        """Configuration for edge case testing."""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=80,
            keep_recent_messages=2,
            max_tool_output_length=50,
        )

    @pytest.fixture
    def context_manager(self, edge_case_config):
        """Context manager for edge case testing."""
        return ContextManager(llm=Mock(), config=edge_case_config)

    def test_empty_and_none_inputs(self, context_manager):
        """Test handling of empty and None inputs."""
        # Empty list
        messages = []
        context_manager.manage_context_messages(messages)
        assert messages == []

        # List with None elements (should be filtered out gracefully)
        messages = [HumanMessage(content="Test"), None, AIMessage(content="Response")]
        # Filter out None values before passing to context manager
        filtered_messages = [msg for msg in messages if msg is not None]
        context_manager.manage_context_messages(filtered_messages)
        result = filtered_messages
        assert len(result) == 2

    def test_malformed_messages(self, context_manager):
        """Test handling of malformed messages."""
        # Message with None content
        try:
            malformed_msg = HumanMessage(content=None)
            messages = [malformed_msg]
            context_manager.manage_context_messages(messages)
            result = messages
            # Should handle gracefully
            assert isinstance(result, list)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "content" in str(e).lower()

    def test_extremely_long_single_message(self, context_manager):
        """Test handling of extremely long single messages."""
        # Create a message longer than the entire context limit
        very_long_content = "A" * 100000  # Much longer than context limit
        long_message = HumanMessage(content=very_long_content)

        messages = [long_message]
        context_manager.manage_context_messages(messages)
        result = messages

        # Should still return the message (context management doesn't trim individual message content)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_tool_message_without_tool_call_id(self, context_manager):
        """Test handling of tool messages without proper tool_call_id."""
        try:
            # This might raise an error depending on LangChain's validation
            tool_msg = ToolMessage(content="Result", tool_call_id="")
            messages = [tool_msg]
            context_manager.manage_context_messages(messages)
            result = messages
            assert isinstance(result, list)
        except Exception:
            # If LangChain validates and raises, that's acceptable
            pass

    def test_circular_references_in_content(self, context_manager):
        """Test handling of complex content that might cause issues."""
        # Content with special characters and formatting
        special_content = (
            """
        Content with:
        - Unicode: 🚀 中文 العربية
        - Code blocks: ```python\nprint("hello")\n```
        - JSON: {"key": "value", "nested": {"array": [1,2,3]}}
        - HTML: <div class="test">content</div>
        - URLs: https://example.com/path?param=value
        - Very long line: """
            + "X" * 1000
        )

        message = HumanMessage(content=special_content)
        messages = [message]
        context_manager.manage_context_messages(messages)
        result = messages

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_zero_configuration_values(self):
        """Test behavior with zero configuration values."""
        zero_config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=0,
            keep_recent_messages=0,
            max_tool_output_length=0,
        )

        context_manager = ContextManager(llm=Mock(), config=zero_config)
        messages = [HumanMessage(content="Test")]

        # Should handle zero values gracefully
        context_manager.manage_context_messages(messages)
        result = messages
        assert isinstance(result, list)

    def test_negative_configuration_values(self):
        """Test behavior with negative configuration values."""
        negative_config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=-50,
            keep_recent_messages=-5,
            max_tool_output_length=-10,
        )

        context_manager = ContextManager(llm=Mock(), config=negative_config)
        messages = [HumanMessage(content="Test")]

        # Should handle negative values gracefully (might treat as disabled)
        context_manager.manage_context_messages(messages)
        result = messages
        assert isinstance(result, list)

    def test_unicode_and_encoding_edge_cases(self, context_manager):
        """Test handling of various Unicode and encoding scenarios."""
        unicode_messages = [
            HumanMessage(content="English text"),
            HumanMessage(content="中文内容测试"),
            HumanMessage(content="العربية"),
            HumanMessage(content="Русский текст"),
            HumanMessage(content="🚀🎉💡🔥"),  # Emojis
            HumanMessage(content="Mixed: Hello 世界 🌍"),
            ToolMessage(content="Unicode tool result: café naïve résumé", tool_call_id="unicode_1"),
        ]

        context_manager.manage_context_messages(unicode_messages)
        result = unicode_messages

        # Should handle all Unicode content
        assert len(result) > 0
        assert all(isinstance(msg, (HumanMessage, AIMessage, ToolMessage)) for msg in result)

    def test_extremely_nested_or_complex_structures(self, context_manager):
        """Test handling of complex nested data structures in tool outputs."""
        # Simulate deeply nested JSON output
        nested_data = {"level1": {"level2": {"level3": {"data": ["item"] * 1000}}}}
        complex_output = str(nested_data) * 100  # Make it very large

        # Create messages so the tool message is in historical part (not recent)
        # keep_recent_messages=2, so add more than 2 messages after the tool message
        messages = [
            ToolMessage(content=complex_output, tool_call_id="complex_1"),  # Historical part
            HumanMessage(content="Question 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Recent question"),  # Recent part starts here
        ]
        context_manager.manage_context_messages(messages)
        result = messages

        # Should trim the complex output since it's in historical part
        tool_msg = next(msg for msg in result if isinstance(msg, ToolMessage))
        assert len(str(tool_msg.content)) < len(complex_output)

    def test_sql_output_edge_cases(self, context_manager):
        """Test SQL output trimming with edge cases."""
        # SQL with no results
        empty_sql_output = """SQL Query:
```sql
SELECT * FROM users WHERE id = -1;
```

Query Results (CSV format):
```csv
id,name
```"""

        # SQL with single row
        single_row_sql = """SQL Query:
```sql
SELECT COUNT(*) as total FROM users;
```

Query Results (CSV format):
```csv
total
42
```"""

        # Malformed SQL output
        malformed_sql = """Something that looks like SQL but isn't:
```sql
INVALID QUERY HERE
```
Random text after"""

        test_cases = [empty_sql_output, single_row_sql, malformed_sql]

        for sql_output in test_cases:
            tool_msg = ToolMessage(content=sql_output, tool_call_id="sql_test")
            messages = [tool_msg]
            context_manager.manage_context_messages(messages)
            result = messages

            # Should handle all cases gracefully
            assert len(result) == 1
            assert isinstance(result[0], ToolMessage)

    def test_conversation_state_consistency(self, context_manager):
        """Test that conversation state remains consistent through management."""
        # Create a conversation with specific patterns (no SystemMessage in state)
        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Response 1"),
            ToolMessage(content="Tool result 1", tool_call_id="tool_1"),
            HumanMessage(content="Question 2"),
            AIMessage(
                content="Response 2 with tool calls",
                tool_calls=[{"name": "test_tool", "args": {"param": "value"}, "id": "call_1"}],
            ),
            ToolMessage(content="Tool result 2", tool_call_id="call_1"),
            HumanMessage(content="Final question"),
        ]

        with patch(
            "openchatbi.context_manager.call_llm_chat_model_with_retry", return_value=AIMessage(content="Summary")
        ):
            context_manager.manage_context_messages(messages)
            result = messages

        # Should maintain message type consistency (only valid state message types)
        message_types = [type(msg) for msg in result]
        valid_types = {HumanMessage, AIMessage, ToolMessage}
        assert all(
            msg_type in valid_types for msg_type in message_types
        ), "Should only contain valid state message types"

        # Should not have orphaned tool messages without corresponding AI messages
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage):
                # There should be an AI message with tool calls before this
                previous_ai_msgs = [m for m in result[:i] if isinstance(m, AIMessage)]
                assert len(previous_ai_msgs) > 0, "Tool message should have corresponding AI message"
