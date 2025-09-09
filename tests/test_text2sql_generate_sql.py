"""Tests for text2sql SQL generation functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from openchatbi.graph_state import SQLGraphState
from openchatbi.text2sql.generate_sql import create_sql_nodes, should_execute_sql, should_retry_sql


class TestText2SQLGenerateSQL:
    """Test text2sql SQL generation functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="SELECT * FROM users")
        return llm

    @pytest.fixture
    def mock_catalog(self):
        """Mock catalog store for testing."""
        catalog = Mock()
        catalog.get_table_information.return_value = {
            "description": "User data table",
            "sql_rule": "",
            "derived_metric": "",
        }
        catalog.get_column_list.return_value = [
            {
                "column_name": "user_id",
                "type": "bigint",
                "display_name": "User ID",
                "description": "Unique user identifier",
                "alias": "",
            }
        ]
        # Mock SQL engine with proper context manager
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [("1", "John"), ("2", "Jane")]
        mock_result.keys.return_value = ["id", "name"]
        mock_connection.execute.return_value = mock_result

        # Create a proper context manager mock using MagicMock
        from unittest.mock import MagicMock

        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_connection
        mock_context_manager.__exit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        catalog.get_sql_engine.return_value = mock_engine

        return catalog

    def test_create_sql_nodes(self, mock_llm, mock_catalog):
        """Test creating SQL processing nodes."""
        generate_node, execute_node, regenerate_node = create_sql_nodes(mock_llm, mock_catalog, "presto")

        assert callable(generate_node)
        assert callable(execute_node)
        assert callable(regenerate_node)

    def test_generate_sql_node_success(self, mock_llm, mock_catalog):
        """Test successful SQL generation."""
        generate_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.get_relevant_documents.return_value = []

            result = generate_node(state)

        assert "sql" in result
        assert result["sql"] == "SELECT * FROM users"

    def test_generate_sql_node_missing_rewrite_question(self, mock_llm, mock_catalog):
        """Test SQL generation with missing rewrite question."""
        generate_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            # Missing rewrite_question
        )

        result = generate_node(state)
        assert result == {}

    def test_generate_sql_node_missing_tables(self, mock_llm, mock_catalog):
        """Test SQL generation with missing tables."""
        generate_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[], question="Show all users", rewrite_question="Show all users", tables=[]  # Empty tables
        )

        result = generate_node(state)
        assert result == {}

    def test_execute_sql_node_success(self, mock_llm, mock_catalog):
        """Test successful SQL execution."""
        _, execute_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_SUCCESS

        assert result["sql_execution_result"] == SQL_SUCCESS
        assert "data" in result

    def test_execute_sql_node_empty_sql(self, mock_llm, mock_catalog):
        """Test SQL execution with empty SQL."""
        _, execute_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(messages=[], sql="")  # Empty SQL

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_NA

        assert result["sql_execution_result"] == SQL_NA

    def test_execute_sql_node_syntax_error(self, mock_llm, mock_catalog):
        """Test SQL execution with syntax error."""
        _, execute_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        # Mock SQL execution to raise syntax error
        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import ProgrammingError

        mock_connection.execute.side_effect = ProgrammingError("", "", "Syntax error")

        state = SQLGraphState(messages=[], sql="SELECT * FRON users")  # Intentional syntax error

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_SYNTAX_ERROR

        assert result["sql_execution_result"] == SQL_SYNTAX_ERROR
        assert "previous_sql_errors" in result

    def test_regenerate_sql_node_success(self, mock_llm, mock_catalog):
        """Test successful SQL regeneration."""
        _, _, regenerate_node = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
            previous_sql_errors=[
                {"sql": "SELECT * FRON users", "error": "Syntax error: FRON", "error_type": "SQL syntax error"}
            ],
            sql_retry_count=1,
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.get_relevant_documents.return_value = []

            result = regenerate_node(state)

        assert "sql" in result
        assert "sql_retry_count" in result
        assert result["sql_retry_count"] == 2

    def test_should_retry_sql_success(self):
        """Test retry decision with successful execution."""
        # Import the constant from the module
        from openchatbi.constants import SQL_SUCCESS

        state = SQLGraphState(sql_execution_result=SQL_SUCCESS, sql_retry_count=1)

        result = should_retry_sql(state)
        assert result == "end"

    def test_should_retry_sql_timeout(self):
        """Test retry decision with timeout."""
        # Import the constant from the module
        from openchatbi.constants import SQL_EXECUTE_TIMEOUT

        state = SQLGraphState(sql_execution_result=SQL_EXECUTE_TIMEOUT, sql_retry_count=1)

        result = should_retry_sql(state)
        assert result == "end"

    def test_should_retry_sql_retry_needed(self):
        """Test retry decision when retry is needed."""
        state = SQLGraphState(sql_execution_result="SYNTAX_ERROR", sql_retry_count=1)

        result = should_retry_sql(state)
        assert result == "regenerate_sql"

    def test_should_retry_sql_max_retries_reached(self):
        """Test retry decision when max retries reached."""
        state = SQLGraphState(sql_execution_result="SYNTAX_ERROR", sql_retry_count=3)

        result = should_retry_sql(state)
        assert result == "end"

    def test_should_execute_sql_with_sql(self):
        """Test execute decision with SQL present."""
        state = SQLGraphState(sql="SELECT * FROM users")

        result = should_execute_sql(state)
        assert result == "execute_sql"

    def test_should_execute_sql_without_sql(self):
        """Test execute decision without SQL."""
        state = SQLGraphState(sql="")

        result = should_execute_sql(state)
        assert result == "end"

    def test_sql_generation_with_examples(self, mock_llm, mock_catalog):
        """Test SQL generation with relevant examples."""
        generate_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show user count",
            rewrite_question="Show user count",
            tables=[{"table": "users", "columns": []}],
        )

        # Mock example retrieval
        mock_document = Mock()
        mock_document.page_content = "How many users are there?"

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.get_relevant_documents.return_value = [mock_document]

            with patch(
                "openchatbi.text2sql.generate_sql.sql_example_dicts",
                {"How many users are there?": ("SELECT COUNT(*) FROM users", ["users"])},
            ):
                result = generate_node(state)

        assert "sql" in result

    def test_sql_error_handling_database_error(self, mock_llm, mock_catalog):
        """Test handling of database connection errors."""
        _, execute_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        # Mock database connection error
        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import OperationalError

        mock_connection.execute.side_effect = OperationalError("", "", "Connection failed")

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_EXECUTE_TIMEOUT

        assert result["sql_execution_result"] == SQL_EXECUTE_TIMEOUT

    def test_regenerate_sql_empty_response(self, mock_llm, mock_catalog):
        """Test regeneration with empty LLM response."""
        mock_llm.invoke.return_value = AIMessage(content="")

        _, _, regenerate_node = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
            previous_sql_errors=[],
            sql_retry_count=1,
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.get_relevant_documents.return_value = []

            result = regenerate_node(state)

        assert "sql" in result
        assert result["sql"] == ""
        from openchatbi.constants import SQL_NA

        assert result["sql_execution_result"] == SQL_NA
