"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine

from openchatbi.catalog.store.file_system import FileSystemCatalogStore
from openchatbi.graph_state import AgentState


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Test configuration fixture."""
    return {
        "organization": "TestOrg",
        "dialect": "presto",
        "bi_config_file": "test_bi.yaml",
        "catalog_store": {"store_type": "file_system", "data_path": "./test_data"},
        "default_llm": {
            "class": "langchain_core.language_models.FakeListChatModel",
            "params": {"responses": ["Test response"]},
        },
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_llm() -> FakeListChatModel:
    """Mock LLM fixture for testing."""
    return FakeListChatModel(
        responses=["SELECT COUNT(*) FROM test_table;", "This is a test SQL query.", "Test analysis result."]
    )


@pytest.fixture
def sample_agent_state() -> AgentState:
    """Sample agent state for testing."""
    return AgentState(
        messages=[HumanMessage(content="Test query")],
        sql="SELECT * FROM test_table;",
        agent_next_node="sql_generation",
        final_answer="Test data results",
    )


@pytest.fixture
def mock_catalog_store(temp_dir: Path) -> FileSystemCatalogStore:
    """Mock catalog store fixture."""
    # Create test data files
    test_data_dir = temp_dir / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    # Create sample table_columns.csv
    tables_info_file = test_data_dir / "table_info.yaml"
    tables_info_file.write_text(
        """test:
  test_table:
    type: fact
    description: A test table for unit tests
  user_data:
    type: fact
    description: User information table"""
    )

    # Create sample table_columns.csv
    tables_file = test_data_dir / "table_columns.csv"
    tables_file.write_text(
        """db_name,table_name,column_name
test,test_table,id
test,test_table,name
test,user_data,user_id"""
    )

    # Create sample table_spec_columns.csv
    columns_file = test_data_dir / "table_spec_columns.csv"
    columns_file.write_text(
        """db_name,table_name,column_name,type,display_name,description
test,test_table,id,bigint,Id,Primary key
test,test_table,name,varchar,Name,User name
test,user_data,user_id,bigint,User Id,User identifier"""
    )

    # Create sample common_columns.csv
    common_columns_file = test_data_dir / "common_columns.csv"
    common_columns_file.write_text(
        """column_name,type,display_name,description
status,varchar,Status,Record status
created_at,timestamp,Created At,Creation timestamp
updated_at,timestamp,Updated At,Last update timestamp"""
    )

    # Mock data warehouse config
    data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}

    return FileSystemCatalogStore(data_path=str(test_data_dir), data_warehouse_config=data_warehouse_config)


@pytest.fixture
def mock_database_engine():
    """Mock database engine fixture."""
    engine = create_engine("sqlite:///:memory:")

    # Create test tables
    with engine.connect() as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test_table VALUES (1, 'Test User')")
        conn.commit()

    return engine


@pytest.fixture
def sample_table_info() -> dict[str, Any]:
    """Sample table information fixture."""
    return {
        "test_table": {
            "columns": [
                {"name": "id", "type": "bigint", "description": "Primary key"},
                {"name": "name", "type": "varchar", "description": "User name"},
            ],
            "description": "A test table for unit tests",
            "sql_rule": "Always filter by active status",
        }
    }


@pytest.fixture
def sample_messages() -> list:
    """Sample message history fixture."""
    return [
        HumanMessage(content="What's the user count?"),
        AIMessage(content="I'll help you get the user count from the database."),
        HumanMessage(content="Show me the SQL query"),
    ]


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, temp_dir):
    """Setup test environment variables."""
    monkeypatch.setenv("OPENCHATBI_CONFIG_PATH", str(temp_dir / "config.yaml"))
    monkeypatch.setenv("OPENCHATBI_TEST_MODE", "true")


class MockTokenService:
    """Mock token service for testing."""

    def __init__(self):
        self.token = "mock_token_12345"

    def get_token(self) -> str:
        return self.token


@pytest.fixture
def mock_token_service() -> MockTokenService:
    """Mock token service fixture."""
    return MockTokenService()


@pytest.fixture
def sample_sql_examples() -> list:
    """Sample SQL examples fixture."""
    return [
        {"question": "How many users are there?", "sql": "SELECT COUNT(*) FROM users;", "tables": ["users"]},
        {
            "question": "What's the average age?",
            "sql": "SELECT AVG(age) FROM users WHERE age IS NOT NULL;",
            "tables": ["users"],
        },
    ]


@pytest.fixture
def mock_presto_connection():
    """Mock Presto connection fixture."""
    mock_conn = Mock()
    mock_cursor = Mock()

    # Setup cursor behavior
    mock_cursor.fetchall.return_value = [("table1", "Test table 1"), ("table2", "Test table 2")]
    mock_cursor.description = [("table_name",), ("description",)]

    mock_conn.cursor.return_value = mock_cursor
    mock_conn.execute.return_value = mock_cursor

    return mock_conn
