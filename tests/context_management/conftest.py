"""Pytest configuration and fixtures for context management tests."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from openchatbi.context_config import ContextConfig


@pytest.fixture
def mock_llm():
    """Mock LLM for testing across all test modules."""
    llm = Mock()
    llm.bind_tools = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_llm_with_summary_response():
    """Mock LLM that returns a summary response."""
    llm = Mock()
    llm.bind_tools = Mock(return_value=llm)
    return llm


@pytest.fixture
def standard_config():
    """Standard test configuration."""
    return ContextConfig(
        enabled=True,
        summary_trigger_tokens=12000,
        keep_recent_messages=10,
        max_tool_output_length=2000,
        max_sql_result_rows=20,
        max_code_output_lines=50,
        enable_summarization=True,
        enable_conversation_summary=True,
        preserve_tool_errors=True,
    )


@pytest.fixture
def minimal_config():
    """Minimal test configuration."""
    return ContextConfig(
        enabled=True,
        summary_trigger_tokens=800,
        keep_recent_messages=3,
        max_tool_output_length=200,
        max_sql_result_rows=5,
        max_code_output_lines=10,
    )


@pytest.fixture
def disabled_config():
    """Configuration with context management disabled."""
    return ContextConfig(
        enabled=False, summary_trigger_tokens=12000, keep_recent_messages=10, max_tool_output_length=2000
    )


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    from langchain_core.messages import HumanMessage, ToolMessage

    return [
        HumanMessage(content="Can you analyze our sales data?"),
        AIMessage(content="I'll help you analyze the sales data. Let me query the database."),
        ToolMessage(content="Query executed successfully. Found 1000 records.", tool_call_id="query_1"),
        HumanMessage(content="What are the top trends?"),
        AIMessage(content="Based on the data, I can see several key trends..."),
        HumanMessage(content="Can you create a visualization?"),
        AIMessage(
            content="I'll create a chart for you.",
            tool_calls=[{"name": "create_chart", "args": {"type": "bar"}, "id": "chart_1"}],
        ),
        ToolMessage(content="Chart created successfully.", tool_call_id="chart_1"),
    ]


@pytest.fixture
def large_sql_output():
    """Large SQL output for testing trimming."""
    csv_data = "id,name,value,date\n"
    csv_data += "\n".join([f"{i},Customer_{i},{i*100},2023-01-{i%30+1:02d}" for i in range(100)])

    return f"""SQL Query:
```sql
SELECT id, name, value, date
FROM customers
ORDER BY value DESC
LIMIT 100;
```

Query Results (CSV format):
```csv
{csv_data}
```

Visualization Created: bar chart has been automatically generated and will be displayed in the UI."""


@pytest.fixture
def large_python_output():
    """Large Python code execution output."""
    output_lines = []
    output_lines.append("Processing data...")
    for i in range(100):
        output_lines.append(f"Step {i}: Processing record {i} - Status: OK")
    output_lines.append("Processing complete!")

    return "\n".join(output_lines)


@pytest.fixture
def error_output():
    """Sample error output."""
    return """Traceback (most recent call last):
  File "/app/analysis.py", line 42, in analyze_sales
    df = pd.read_csv('nonexistent_file.csv')
  File "/usr/local/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 912, in read_csv
    return _read(filepath_or_buffer, kwds)
FileNotFoundError: [Errno 2] No such file or directory: 'nonexistent_file.csv'

Error: Could not load the sales data file. Please check that the file exists and is accessible."""


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests based on certain patterns
        if any(pattern in item.nodeid.lower() for pattern in ["large", "stress", "concurrent"]):
            item.add_marker(pytest.mark.slow)
