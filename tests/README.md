# OpenChatBI Test Suite

This directory contains comprehensive unit tests for the OpenChatBI project. The test suite is built using pytest and follows modern Python testing best practices.

## Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared fixtures and configuration
├── README.md                            # This file
│
├── Core Module Tests
├── test_config_loader.py                # Configuration loading tests
├── test_graph_state.py                  # State management tests
├── test_utils.py                        # Utility function tests
│
├── Catalog System Tests
├── test_catalog_store.py                # Catalog store interface tests
├── test_catalog_loader.py               # Database catalog loading tests
│
├── Text2SQL Pipeline Tests
├── test_text2sql_extraction.py          # Information extraction tests
├── test_text2sql_generate_sql.py        # SQL generation tests
├── test_text2sql_schema_linking.py      # Schema linking tests
│
└── Tool Tests
    ├── test_tools_ask_human.py          # Human interaction tool tests
    ├── test_tools_run_python_code.py    # Python code execution tests
    └── test_tools_search_knowledge.py   # Knowledge search tests
```

## Running Tests

### Prerequisites

Ensure you have the development dependencies installed:

```bash
# Using uv (recommended)
uv sync --group dev

# Or using pip
pip install -e ".[dev]"
```

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_config_loader.py

# Run specific test class
uv run pytest tests/test_config_loader.py::TestConfigLoader

# Run specific test method
uv run pytest tests/test_config_loader.py::TestConfigLoader::test_load_config_from_file
```

### Test Coverage

```bash
# Run tests with coverage report
uv run pytest --cov=openchatbi --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

### Test Categories

```bash
# Run only fast unit tests (exclude slow integration tests)
uv run pytest -m "not slow"

# Run tests for specific components
uv run pytest tests/test_catalog* -k "catalog"
uv run pytest tests/test_text2sql* -k "text2sql"
uv run pytest tests/test_tools* -k "tools"
```

## Test Configuration

### Environment Variables

The test suite uses several environment variables that can be set to customize test behavior:

- `OPENCHATBI_TEST_MODE=true` - Enables test mode
- `OPENCHATBI_CONFIG_PATH` - Path to test configuration file
- `PYTEST_TIMEOUT=300` - Test timeout in seconds

### Fixtures

The `conftest.py` file provides shared fixtures used across tests:

#### Core Fixtures
- `test_config` - Test configuration dictionary
- `temp_dir` - Temporary directory for test files
- `mock_llm` - Mocked language model for testing
- `sample_agent_state` - Sample AgentState for testing

#### Catalog Fixtures
- `mock_catalog_store` - Mocked catalog store with sample data
- `mock_database_engine` - Mocked database engine
- `sample_table_info` - Sample table metadata

#### Database Fixtures
- `mock_presto_connection` - Mocked Presto database connection
- `mock_token_service` - Mocked authentication token service

## Writing Tests

### Test Naming Conventions

Follow these naming conventions for consistency:

```python
# Test files
test_<module_name>.py

# Test classes
class TestModuleName:

# Test methods
def test_specific_functionality(self):
def test_error_condition_handling(self):
def test_edge_case_scenario(self):
```

### Test Categories

Use pytest marks to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_basic_functionality():
    """Unit test for basic functionality."""
    pass

@pytest.mark.integration
def test_database_integration():
    """Integration test with database."""
    pass

@pytest.mark.slow
def test_performance_benchmark():
    """Slow performance test."""
    pass

@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
])
def test_multiple_scenarios(input, expected):
    """Test multiple input/output scenarios."""
    pass
```

### Mocking Best Practices

Use proper mocking for external dependencies:

```python
from unittest.mock import Mock, patch, MagicMock

# Mock external services
@patch('openchatbi.module.external_service')
def test_with_external_service(mock_service):
    mock_service.return_value = "expected_result"
    # Test implementation

# Mock LLM responses
def test_llm_integration(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="Mock response")
    # Test implementation
```

### Async Test Support

For testing async functionality:

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_functionality():
    """Test asynchronous operations."""
    result = await some_async_function()
    assert result is not None
```

## Test Data

### Sample Data Files

Test data is managed through fixtures and temporary files:

```python
def test_with_sample_data(temp_dir):
    """Test using temporary sample data."""
    # Create test data file
    data_file = temp_dir / "test_data.csv"
    data_file.write_text("col1,col2\\nval1,val2")
    
    # Test with the data
    assert data_file.exists()
```

### Mock Responses

Common mock responses are defined in fixtures:

```python
# SQL generation mock response
mock_llm.invoke.return_value = AIMessage(
    content="SELECT COUNT(*) FROM test_table;"
)

# Catalog search mock response
mock_catalog.search_tables.return_value = [
    {"table_name": "users", "description": "User data"}
]
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled runs (daily)

### Test Matrix

Tests run against multiple configurations:
- Python versions: 3.11+
- Dependencies: Minimum and latest versions

## Debugging Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install -e ".[test]"
   ```

3. **Configuration Issues**
   ```bash
   # Set test environment variables
   export OPENCHATBI_TEST_MODE=true
   ```

### Debug Output

Enable debug output for failing tests:

```bash
# Run with debug output
uv run pytest -v -s --tb=long

# Run with pdb on failures
uv run pytest --pdb

# Run with coverage debug
uv run pytest --cov-report=term-missing -v
```

## Performance Testing

### Benchmarks

Performance tests are marked with `@pytest.mark.slow`:

```bash
# Run performance tests
uv run pytest -m slow

# Skip performance tests
uv run pytest -m "not slow"
```

### Memory Profiling

For memory usage testing:

```bash
# Install memory profiler
pip install memory-profiler

# Run with memory profiling
uv run pytest --profile-mem
```

## Contributing

### Adding New Tests

1. Create test file following naming conventions
2. Import required fixtures from `conftest.py`
3. Write comprehensive test cases covering:
   - Happy path scenarios
   - Error conditions
   - Edge cases
   - Performance considerations

4. Use appropriate mocking for external dependencies
5. Add docstrings explaining test purpose
6. Run tests locally before submitting PR

### Test Review Guidelines

When reviewing test PRs:
- Ensure adequate test coverage
- Verify mock usage is appropriate
- Check for test independence
- Validate error case handling
- Confirm performance test categorization

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)