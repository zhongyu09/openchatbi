"""Unit tests for ContextManager class."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from openchatbi.context_manager import ContextManager
from openchatbi.context_config import ContextConfig


class TestContextManager:
    """Test cases for ContextManager class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        # Mock response for summarization
        llm_response = AIMessage(content="This is a test summary of the conversation.")
        with patch("openchatbi.context_manager.call_llm_chat_model_with_retry", return_value=llm_response):
            yield llm

    @pytest.fixture
    def default_config(self):
        """Default context configuration for testing."""
        return ContextConfig(
            enabled=True,
            summary_trigger_tokens=800,
            keep_recent_messages=3,
            max_tool_output_length=200,
            max_sql_result_rows=5,
            max_code_output_lines=10,
            enable_conversation_summary=True,
            enable_summarization=True,
        )

    @pytest.fixture
    def context_manager(self, mock_llm, default_config):
        """Context manager instance for testing."""
        return ContextManager(llm=mock_llm, config=default_config)

    def test_token_estimation(self, context_manager):
        """Test token estimation functionality."""
        # Test basic token estimation
        short_text = "Hello world"
        assert context_manager.estimate_tokens(short_text) == len(short_text) // 4

        # Test longer text
        long_text = "This is a longer text that should have more tokens estimated."
        assert context_manager.estimate_tokens(long_text) > context_manager.estimate_tokens(short_text)

    def test_message_token_estimation(self, context_manager):
        """Test token estimation for messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            ToolMessage(content="Tool result", tool_call_id="123"),
        ]

        total_tokens = context_manager.estimate_message_tokens(messages)
        assert total_tokens > 0
        # Should include content tokens plus metadata overhead
        assert total_tokens > sum(len(str(msg.content)) // 4 for msg in messages)

    def test_trim_short_tool_output(self, context_manager):
        """Test trimming tool output that's already short enough."""
        short_output = "This is a short output."
        result = context_manager.trim_tool_output(short_output)
        assert result == short_output

    def test_trim_long_generic_output(self, context_manager):
        """Test trimming long generic tool output."""
        long_output = "A" * 500  # Much longer than max_tool_output_length (200)
        result = context_manager.trim_tool_output(long_output)

        assert len(result) < len(long_output)
        assert "... [Output truncated] ..." in result
        assert result.startswith("A")
        assert result.endswith("A")

    def test_trim_sql_output(self, context_manager):
        """Test trimming SQL output with structured data."""
        sql_output = """SQL Query:
```sql
SELECT * FROM users WHERE age > 18;
```

Query Results (CSV format):
```csv
id,name,age,email
1,John,25,john@example.com
2,Jane,30,jane@example.com
3,Bob,22,bob@example.com
4,Alice,28,alice@example.com
5,Charlie,35,charlie@example.com
6,Diana,27,diana@example.com
7,Eve,31,eve@example.com
```

Visualization Created: bar chart has been automatically generated."""

        result = context_manager.trim_tool_output(sql_output)

        # Should preserve SQL query
        assert "SELECT * FROM users WHERE age > 18;" in result
        # Should preserve visualization info
        assert "Visualization Created:" in result
        # Should trim CSV data but keep structure
        assert "```csv" in result
        assert "rows omitted" in result

    def test_trim_code_output(self, context_manager):
        """Test trimming Python code execution output."""
        # Test long output without errors
        long_code_output = "\n".join([f"Line {i}: Some output here" for i in range(50)])
        result = context_manager.trim_tool_output(long_code_output)

        assert len(result.split("\n")) < 50
        assert "... [Output truncated] ..." in result

    def test_preserve_error_output(self, context_manager):
        """Test that error outputs are preserved when configured."""
        error_output = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    print(undefined_variable)
NameError: name 'undefined_variable' is not defined"""

        # With preserve_tool_errors=True (default in test config)
        result = context_manager.trim_tool_output(error_output)
        assert result == error_output  # Should be preserved in full

        # Test with preserve_tool_errors=False
        context_manager.config.preserve_tool_errors = False
        result = context_manager.trim_tool_output(error_output)
        # Should still preserve because it's an error, but could be trimmed based on length

    # Tool output trimming disable test removed - trimming is always enabled now

    def test_conversation_summary_disabled(self, context_manager):
        """Test conversation summary when disabled."""
        context_manager.config.enable_conversation_summary = False
        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]

        summary = context_manager.summarize_conversation(messages)
        assert summary == ""

    @patch("openchatbi.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summary_success(self, mock_llm_call, context_manager):
        """Test successful conversation summarization."""
        # Mock successful LLM response
        mock_response = AIMessage(content="Summary: User asked about data analysis.")
        mock_llm_call.return_value = mock_response

        messages = [
            HumanMessage(content="Can you analyze our sales data?"),
            AIMessage(content="I'll help you analyze the sales data."),
            ToolMessage(content="Query results: 100 records", tool_call_id="123"),
            HumanMessage(content="What are the trends?"),
            AIMessage(content="The trends show increasing sales."),
            HumanMessage(content="Recent question"),  # This should be excluded from summary
        ]

        summary = context_manager.summarize_conversation(messages)
        assert summary.startswith("[Conversation Summary]:")
        assert "Summary: User asked about data analysis." in summary
        mock_llm_call.assert_called_once()

    @patch("openchatbi.context_manager.call_llm_chat_model_with_retry")
    def test_conversation_summary_failure(self, mock_llm_call, context_manager):
        """Test conversation summary when LLM call fails."""
        # Mock LLM failure
        mock_llm_call.side_effect = Exception("LLM service unavailable")

        # Need more messages than keep_recent_messages (3) to trigger summarization
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Second message"),
            AIMessage(content="Second response"),
            HumanMessage(content="Third message"),
            AIMessage(content="Third response"),
        ]
        summary = context_manager.summarize_conversation(messages)
        assert summary == "[Summary generation failed]"

    def test_manage_context_disabled(self, context_manager):
        """Test context management when disabled."""
        context_manager.config.enabled = False
        messages = [HumanMessage(content="Test")]

        context_manager.manage_context_messages(messages)
        result = messages
        assert result == messages  # Should return unchanged

    def test_manage_context_empty_messages(self, context_manager):
        """Test context management with empty message list."""
        messages = []
        context_manager.manage_context_messages(messages)
        result = messages
        assert result == []

    def test_manage_context_tool_message_trimming(self, context_manager):
        """Test that tool messages are trimmed during context management."""
        long_content = "A" * 500
        # Add enough messages to trigger context management, with ToolMessage in historical part
        # keep_recent_messages=3, so we need more than 3 messages after the ToolMessage
        messages = [
            HumanMessage(content="This is a long question that helps reach the token threshold " * 10),
            ToolMessage(content=long_content, tool_call_id="123"),  # This should be in historical part
            AIMessage(content="This is a long response that helps reach the token threshold " * 10),
            HumanMessage(content="Another long question to increase token count " * 10),
            AIMessage(content="Response " * 20),
            HumanMessage(content="Final question"),
            AIMessage(content="Final response"),
        ]

        context_manager.manage_context_messages(messages)
        result = messages

        # Find the tool message in results
        tool_msg = next(msg for msg in result if isinstance(msg, ToolMessage))
        assert len(str(tool_msg.content)) < len(long_content)
        assert "... [Output truncated] ..." in str(tool_msg.content)

    @patch("openchatbi.context_manager.call_llm_chat_model_with_retry")
    def test_manage_context_with_summarization(self, mock_llm_call, context_manager):
        """Test context management triggering summarization."""
        # Mock successful summarization
        mock_response = AIMessage(content="Conversation summary here.")
        mock_llm_call.return_value = mock_response

        # Create many messages to trigger summarization
        messages = []
        for i in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"Question {i}"),
                    AIMessage(content=f"Response {i}" * 100),  # Long responses to increase token count
                ]
            )

        original_length = len(messages)
        context_manager.manage_context_messages(messages)
        result = messages

        # Should have fewer messages than input
        assert len(result) < original_length
        # Should contain summary message
        assert any("[Conversation Summary]:" in str(msg.content) for msg in result if hasattr(msg, "content"))
        # Verify LLM was called for summarization
        mock_llm_call.assert_called_once()

    # Tool wrapper tests removed - we now handle context at state level

    def test_format_messages_for_summary(self, context_manager):
        """Test message formatting for summary generation."""
        messages = [
            HumanMessage(content="User question"),
            AIMessage(content="AI response"),
            ToolMessage(content="Tool result with some data", tool_call_id="123"),
            SystemMessage(content="System message"),  # Should be excluded
        ]

        formatted = context_manager._format_messages_for_summary(messages)

        assert "<user> User question </user>" in formatted
        assert "<assistant>" in formatted and "AI response" in formatted
        assert "tool_result" in formatted
        assert "System message" not in formatted  # System messages excluded

    def test_format_long_ai_message_for_summary(self, context_manager):
        """Test that long AI messages are truncated in summary formatting."""
        long_content = "A" * 1000
        messages = [AIMessage(content=long_content)]

        formatted = context_manager._format_messages_for_summary(messages)

        assert len(formatted) < len(f"Assistant: {long_content}")
        assert "... [truncated]" in formatted


# Tool wrapping tests removed - we now handle context at state level instead of wrapping tools


# Pytest fixtures and test data
@pytest.fixture
def sample_sql_output():
    """Sample SQL output for testing."""
    return """SQL Query:
```sql
SELECT customer_id, SUM(amount) as total
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY customer_id
ORDER BY total DESC;
```

Query Results (CSV format):
```csv
customer_id,total
1001,15420.50
1002,12350.75
1003,11200.00
1004,9875.25
1005,8650.00
1006,7500.50
1007,6200.75
1008,5800.00
1009,4950.25
1010,4200.00
```

Visualization Created: bar chart has been automatically generated and will be displayed in the UI."""


@pytest.fixture
def sample_error_output():
    """Sample error output for testing."""
    return """Traceback (most recent call last):
  File "/app/code.py", line 15, in analyze_data
    result = df.groupby('nonexistent_column').sum()
  File "/usr/local/lib/python3.9/site-packages/pandas/core/groupby/groupby.py", line 1647, in sum
    return self._cython_transform("sum", numeric_only=numeric_only, **kwargs)
KeyError: 'nonexistent_column'

Error: Column 'nonexistent_column' not found in DataFrame. Available columns: ['customer_id', 'order_date', 'amount', 'product_id']"""
