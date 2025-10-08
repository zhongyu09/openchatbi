# Context Management Test Suite

This directory contains comprehensive tests for the context management functionality in OpenChatBI.

## Test Structure

### 📁 Test Files

- **`test_context_manager.py`** - Unit tests for the `ContextManager` class
- **`test_context_config.py`** - Tests for context configuration management
- **`test_agent_graph_integration.py`** - Integration tests for agent graph with context management
- **`test_edge_cases.py`** - Edge case handling
- **`test_state_operations.py`** - Tests for state operations and message processing
- **`conftest.py`** - Shared pytest fixtures and configuration
- **`test_runner.py`** - Custom test runner script

## 🧪 Test Categories

### Unit Tests (`test_context_manager.py`)

Tests core functionality of the `ContextManager` class:

- ✅ Token estimation and message token calculation
- ✅ Tool output trimming (generic, SQL, Python code)
- ✅ Conversation summarization
- ✅ Context management with sliding window
- ✅ Tool wrapper functionality
- ✅ Configuration-based behavior

**Key test cases:**
- `test_trim_sql_output()` - Tests intelligent SQL result trimming
- `test_conversation_summary_success()` - Tests LLM-based summarization
- `test_manage_context_with_summarization()` - Tests full context management flow

### Configuration Tests (`test_context_config.py`)

Tests configuration management and validation:

- ✅ Default configuration values
- ✅ Custom configuration creation
- ✅ Configuration updates
- ✅ Edge cases (zero/negative values)
- ✅ Different configuration presets

**Key test cases:**
- `test_update_context_config_multiple_values()` - Tests configuration updates
- `test_production_optimized_config()` - Tests realistic production settings

### Integration Tests (`test_agent_graph_integration.py`)

Tests integration with the agent graph system:

- ✅ Agent router with context management
- ✅ Graph building with/without context management
- ✅ Tool wrapping in graph context
- ✅ Full conversation flow testing
- ✅ System message preservation

**Key test cases:**
- `test_agent_router_with_context_manager()` - Tests router integration
- `test_full_context_management_flow()` - Tests end-to-end functionality

### State Operations Tests (`test_state_operations.py`)

Tests state manipulation and message processing operations:

- ✅ Message trimming and truncation logic
- ✅ State updates and modifications
- ✅ Message type handling and conversion
- ✅ Context state preservation during operations
- ✅ Error handling in state operations

**Key test cases:**
- `test_trim_messages_by_token_count()` - Tests message trimming logic
- `test_state_message_processing()` - Tests state message operations
- `test_context_state_updates()` - Tests context state modifications

### Edge Cases (`test_edge_cases.py`)

Tests system behavior under stress and edge conditions:

- ✅ Unicode and encoding edge cases
- ✅ Malformed input handling

**Key test cases:**
- `test_sql_output_edge_cases()` - SQL edge cases
- `test_extremely_nested_or_complex_structures()` - Complex data structures

## 🚀 Running Tests

### Using the Test Runner

```bash
# Run all tests
python tests/context_management/test_runner.py

# Run only unit tests
python tests/context_management/test_runner.py --type unit

# Run with coverage reporting
python tests/context_management/test_runner.py --coverage
```

### Using Pytest Directly

```bash
# Run all context management tests
pytest tests/context_management/

# Run specific test file
pytest tests/context_management/test_context_manager.py

# Run with verbose output
pytest tests/context_management/ -v

# Run with coverage
pytest tests/context_management/ --cov=openchatbi.context_manager --cov-report=html
```

## 📊 Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests (can be excluded)

## 🎯 Test Coverage Areas

### Core Functionality
- [x] Token estimation and management
- [x] Message processing and trimming
- [x] Conversation summarization
- [x] Context compression strategies

### Tool Output Management
- [x] SQL output trimming with structure preservation
- [x] Python code output handling
- [x] Error message preservation
- [x] Generic output trimming

### Configuration Management
- [x] Default and custom configurations
- [x] Configuration validation
- [x] Runtime configuration updates
- [x] Edge case configurations

### Integration Points
- [x] Agent router integration
- [x] Graph building integration
- [x] Tool wrapper integration
- [x] LLM service integration

### Edge Cases
- [x] Unicode and encoding issues
- [x] Malformed input handling

## 🧩 Fixtures

### Common Fixtures (in `conftest.py`)

- `mock_llm` - Mock language model for testing
- `standard_config` - Standard test configuration
- `minimal_config` - Minimal configuration for edge testing
- `sample_conversation` - Sample conversation data
- `large_sql_output` - Large SQL output for trimming tests
- `error_output` - Sample error output for preservation tests

## 🔧 Extending Tests

### Adding New Test Cases

1. Choose the appropriate test file based on the functionality
2. Use existing fixtures from `conftest.py`
3. Follow the naming convention: `test_feature_description()`
4. Add appropriate markers for categorization

### Adding New Fixtures

Add shared fixtures to `conftest.py` if they'll be used across multiple test files.

## 🐛 Debugging Tests

### Common Issues

1. **Mock LLM failures**: Ensure proper mocking of LLM responses
2. **Configuration conflicts**: Use isolated config instances
3. **Memory leaks in large tests**: Force garbage collection with `gc.collect()`

### Debugging Tools

```bash
# Run with debugging output
pytest tests/context_management/ -v -s

# Run single test with full traceback
pytest tests/context_management/test_name.py::test_function -v --tb=long

# Profile test performance
pytest tests/context_management/ --profile
```

## 📋 Test Results

Expected test results:
- **Total tests**: ~100+ test cases across 6 test files
- **Coverage target**: >95% for context management modules
- **State operations tests**: All message processing should work correctly
- **Edge cases**: All should handle gracefully without crashes