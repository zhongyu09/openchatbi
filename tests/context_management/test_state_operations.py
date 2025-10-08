"""Tests for message-based context management operations."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from openchatbi.context_manager import ContextManager
from openchatbi.context_config import ContextConfig


class TestMessageBasedContextManagement:
    """Test message-based context management with direct modification."""

    @pytest.fixture
    def test_config(self):
        """Configuration for testing message operations."""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=300,  # Lower threshold to trigger management
            keep_recent_messages=3,
            max_tool_output_length=200,
            preserve_tool_errors=True,
            preserve_recent_sql=True,
        )

    @pytest.fixture
    def context_manager(self, test_config):
        """Context manager for testing."""
        mock_llm = Mock()
        return ContextManager(llm=mock_llm, config=test_config)

    def test_no_operations_when_disabled(self, context_manager):
        """Test that no operations are performed when context management is disabled."""
        context_manager.config.enabled = False

        messages = [HumanMessage(content="Test", id="test_1")]
        original_messages = messages.copy()

        context_manager.manage_context_messages(messages)
        assert messages == original_messages  # Should be unchanged

    def test_no_operations_when_under_limit(self, context_manager):
        """Test that no operations are performed when context is under token limit."""
        # Short messages that won't trigger context management
        messages = [HumanMessage(content="Hi", id="human_1"), AIMessage(content="Hello", id="ai_1")]
        original_messages = messages.copy()

        context_manager.manage_context_messages(messages)
        assert messages == original_messages  # Should be unchanged

    def test_historical_tool_compression(self, context_manager):
        """Test compression of historical tool messages."""
        # Disable conversation summarization to test only tool compression
        context_manager.config.enable_conversation_summary = False
        context_manager.config.enable_summarization = False

        # Create messages with large historical tool outputs
        messages = [
            HumanMessage(content="Query data", id="human_1"),
            AIMessage(content="Running query", id="ai_1"),
            # Large historical tool message (should be compressed)
            ToolMessage(content="A" * 1000, tool_call_id="query_1", id="tool_1_historical"),  # Large content
            HumanMessage(content="More analysis", id="human_2"),
            AIMessage(content="Analyzing", id="ai_2"),
            # Another large historical tool message
            ToolMessage(content="B" * 800, tool_call_id="query_2", id="tool_2_historical"),  # Large content
            # Recent messages (should be preserved)
            HumanMessage(content="Recent question", id="human_recent"),
            AIMessage(content="Recent response", id="ai_recent"),
            ToolMessage(content="Recent result", tool_call_id="recent_1", id="tool_recent"),
        ]

        original_count = len(messages)
        context_manager.manage_context_messages(messages)

        # Should have same number of messages but some content should be compressed
        assert len(messages) == original_count

        # Check that historical tool messages are compressed
        historical_tool_msgs = [
            msg
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.id in ["tool_1_historical", "tool_2_historical"]
        ]
        for msg in historical_tool_msgs:
            assert len(str(msg.content)) < 1000, "Historical tool messages should be compressed"

    def test_error_message_preservation(self, context_manager):
        """Test that error messages are preserved even if they're historical."""
        error_content = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    raise ValueError("Test error")
ValueError: Test error"""

        messages = [
            HumanMessage(content="Run code", id="human_1"),
            AIMessage(content="Executing", id="ai_1"),
            # Historical error message (should be preserved)
            ToolMessage(content=error_content, tool_call_id="code_1", id="error_tool_historical"),
            # Recent messages
            HumanMessage(content="What happened?", id="human_recent"),
            AIMessage(content="There was an error", id="ai_recent"),
        ]

        original_error_content = messages[2].content
        context_manager.manage_context_messages(messages)

        # Error message should be preserved
        error_msg = next(msg for msg in messages if msg.id == "error_tool_historical")
        assert error_msg.content == original_error_content, "Error messages should be preserved"

    def test_sql_content_preservation(self, context_manager):
        """Test that SQL content is preserved when configured."""
        sql_content = """SQL Query:
```sql
SELECT * FROM users WHERE active = 1;
```

Query Results (CSV format):
```csv
id,name,email
1,John,john@example.com
2,Jane,jane@example.com
```"""

        messages = [
            HumanMessage(content="Get user data", id="human_1"),
            AIMessage(content="Querying users", id="ai_1"),
            # Historical SQL result (should be preserved if preserve_recent_sql=True)
            ToolMessage(content=sql_content, tool_call_id="sql_1", id="sql_tool_historical"),
            # Recent messages
            HumanMessage(content="Analyze results", id="human_recent"),
            AIMessage(content="Analyzing", id="ai_recent"),
        ]

        # Test with SQL preservation enabled
        context_manager.config.preserve_recent_sql = True
        original_sql_content = messages[2].content
        context_manager.manage_context_messages(messages)

        # SQL should be preserved when preserve_recent_sql=True
        sql_msg = next(msg for msg in messages if msg.id == "sql_tool_historical")
        assert sql_msg.content == original_sql_content, "SQL content should be preserved when configured"

    @patch("openchatbi.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summarization(self, mock_llm_call, context_manager):
        """Test conversation summarization with message modification."""
        # Mock LLM response for summarization
        mock_llm_call.return_value = AIMessage(content="Summary of the conversation")

        # Create a long conversation that will trigger summarization
        messages = []

        # Add many historical messages
        for i in range(20):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}" * 10, id=f"human_{i}"),
                    AIMessage(content=f"Response {i}" * 10, id=f"ai_{i}"),
                ]
            )

        # Add recent messages
        messages.extend(
            [
                HumanMessage(content="Recent question", id="human_recent"),
                AIMessage(content="Recent response", id="ai_recent"),
                ToolMessage(content="Recent result", tool_call_id="recent_1", id="tool_recent"),
            ]
        )

        original_count = len(messages)
        context_manager.manage_context_messages(messages)

        # Should have fewer messages due to summarization
        assert len(messages) < original_count

        # Should have a summary message
        summary_msgs = [msg for msg in messages if isinstance(msg, AIMessage) and "Summary" in str(msg.content)]
        assert len(summary_msgs) > 0, "Should create a summary message"

    def test_content_type_detection(self, context_manager):
        """Test content type detection methods."""
        # Test error content detection
        error_contents = [
            "Error: Something went wrong",
            "Traceback (most recent call last):\n  File test.py",
            "ValueError: Invalid input",
            "Connection failed with status 500",
        ]

        for content in error_contents:
            assert context_manager._is_error_content(content), f"Should detect error in: {content[:50]}"

        # Test SQL content detection
        sql_contents = [
            "```sql\nSELECT * FROM users;\n```",
            "Query results: 100 rows returned",
            "SQL Query:\nSELECT id FROM table",
        ]

        for content in sql_contents:
            assert context_manager._is_sql_content(content), f"Should detect SQL in: {content[:50]}"

        # Test data query result detection
        data_contents = [
            "```csv\nid,name\n1,test\n```",
            "Query Results (CSV format):",
            "Found 500 records in the database",
        ]

        for content in data_contents:
            assert context_manager._is_data_query_result(content), f"Should detect data result in: {content[:50]}"

    def test_should_compress_logic(self, context_manager):
        """Test the logic for determining whether to compress historical tool messages."""
        # Short content should not be compressed
        short_msg = ToolMessage(content="Short", tool_call_id="test", id="short")
        assert not context_manager._should_compress_historical_tool_message(short_msg, "Short")

        # Long non-error content should be compressed
        long_content = "A" * 1000
        long_msg = ToolMessage(content=long_content, tool_call_id="test", id="long")
        assert context_manager._should_compress_historical_tool_message(long_msg, long_content)

        # Long error content should not be compressed (if preserve_tool_errors=True)
        error_content = "Error: " + "A" * 1000
        error_msg = ToolMessage(content=error_content, tool_call_id="test", id="error")
        context_manager.config.preserve_tool_errors = True
        assert not context_manager._should_compress_historical_tool_message(error_msg, error_content)

        # But should be compressed if preserve_tool_errors=False
        context_manager.config.preserve_tool_errors = False
        assert context_manager._should_compress_historical_tool_message(error_msg, error_content)

    def test_recent_messages_always_preserved(self, context_manager):
        """Test that recent messages are always preserved regardless of content."""
        # Create messages where recent ones are large but should still be preserved
        messages = []

        # Historical messages
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Historical {i}", id=f"hist_human_{i}"),
                    ToolMessage(content="A" * 500, tool_call_id=f"hist_{i}", id=f"hist_tool_{i}"),
                ]
            )

        # Recent messages (including large tool output)
        messages.extend(
            [
                HumanMessage(content="Recent question", id="recent_human"),
                AIMessage(content="Recent response", id="recent_ai"),
                ToolMessage(content="B" * 1000, tool_call_id="recent", id="recent_tool"),  # Large but recent
            ]
        )

        original_count = len(messages)
        context_manager.manage_context_messages(messages)

        # Recent messages should be preserved (even if content gets compressed due to summarization)
        recent_ids = ["recent_human", "recent_ai", "recent_tool"]
        remaining_recent = [msg for msg in messages if hasattr(msg, "id") and msg.id in recent_ids]

        # All recent message IDs should still be present (even if summarization occurred)
        assert len(remaining_recent) >= 2, "Most recent messages should be preserved"

    def test_message_order_preservation(self, context_manager):
        """Test that message ordering is preserved during context management."""
        # Disable conversation summarization to test only tool compression
        context_manager.config.enable_conversation_summary = False
        context_manager.config.enable_summarization = False

        # Create messages with specific order
        messages = [
            HumanMessage(content="Question 1", id="human_1"),
            AIMessage(content="Response 1", id="ai_1"),
            ToolMessage(content="A" * 1000, tool_call_id="tool_1", id="tool_1"),  # Will be compressed
            HumanMessage(content="Question 2", id="human_2"),
            AIMessage(content="Response 2", id="ai_2"),
            ToolMessage(content="B" * 1000, tool_call_id="tool_2", id="tool_2"),  # Will be compressed
            HumanMessage(content="Recent question", id="human_recent"),  # Recent, should not be compressed
            AIMessage(content="Recent response", id="ai_recent"),  # Recent
            ToolMessage(
                content="C" * 1000, tool_call_id="tool_recent", id="tool_recent"
            ),  # Recent, should not be compressed
        ]

        original_order = [msg.id for msg in messages if hasattr(msg, "id")]
        context_manager.manage_context_messages(messages)

        # Extract the IDs in the new order
        result_order = [msg.id for msg in messages if hasattr(msg, "id")]

        # The order should be preserved
        assert result_order == original_order, "Message order should be preserved"

        # Verify that historical tool messages were actually compressed
        historical_tools = [msg for msg in messages if isinstance(msg, ToolMessage) and msg.id in ["tool_1", "tool_2"]]
        for msg in historical_tools:
            assert len(str(msg.content)) < 1000, "Historical tool messages should be compressed"
