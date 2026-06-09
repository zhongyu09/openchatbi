"""Tests for text2sql SQL generation functionality."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from openchatbi.constants import SQL_EXECUTE_TIMEOUT, SQL_RESULT_LIMIT, SQL_SUCCESS
from openchatbi.graph_state import SQLGraphState
from openchatbi.text2sql.generate_sql import create_sql_nodes, should_execute_sql
from openchatbi.text2sql.sql_graph import _should_generate_visualization_or_retry


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
        mock_result.fetchmany.return_value = [("1", "John"), ("2", "Jane")]
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
        generate_node, execute_node, regenerate_node, visualization_node = create_sql_nodes(
            mock_llm, mock_catalog, "presto"
        )

        assert callable(generate_node)
        assert callable(execute_node)
        assert callable(regenerate_node)
        assert callable(visualization_node)

    def test_generate_sql_node_success(self, mock_llm, mock_catalog):
        """Test successful SQL generation."""
        generate_node, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.invoke.return_value = []

            result = generate_node(state)

        assert "sql" in result
        assert result["sql"] == "SELECT * FROM users"

    def test_generate_sql_node_null_response_does_not_execute(self, mock_llm, mock_catalog):
        """Test NULL SQL generation is treated as no executable SQL."""
        mock_llm.invoke.return_value = AIMessage(content="NULL")
        generate_node, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.invoke.return_value = []
            result = generate_node(state)

        assert result["sql"] == "NULL"
        assert should_execute_sql(result) == "end"

    def test_generate_sql_node_missing_rewrite_question(self, mock_llm, mock_catalog):
        """Test SQL generation with missing rewrite question."""
        generate_node, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            # Missing rewrite_question
        )

        result = generate_node(state)
        assert result == {}

    def test_generate_sql_node_missing_tables(self, mock_llm, mock_catalog):
        """Test SQL generation with missing tables."""
        generate_node, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[], question="Show all users", rewrite_question="Show all users", tables=[]  # Empty tables
        )

        result = generate_node(state)
        assert result == {}

    def test_execute_sql_node_success(self, mock_llm, mock_catalog):
        """Test successful SQL execution."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_SUCCESS

        assert result["sql_execution_result"] == SQL_SUCCESS
        assert "data" in result

    def test_execute_sql_node_applies_result_limit(self, mock_llm, mock_catalog):
        """Test SQL execution limits rows returned to the agent."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        with patch("openchatbi.text2sql.generate_sql.config") as mock_config:
            mock_config.get.side_effect = ValueError
            result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        mock_result = mock_connection.execute.return_value
        executed_sql = str(mock_connection.execute.call_args.args[0])

        assert f"LIMIT {SQL_RESULT_LIMIT}" in executed_sql
        mock_result.fetchmany.assert_called_once_with(SQL_RESULT_LIMIT)
        mock_result.fetchall.assert_not_called()
        assert f"limited to first {SQL_RESULT_LIMIT} rows" in result["messages"][0].content

    def test_execute_sql_node_uses_configured_result_limit(self, mock_llm, mock_catalog):
        """Test SQL execution uses the configured row limit."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        with patch("openchatbi.text2sql.generate_sql.config") as mock_config:
            mock_config.get.return_value = SimpleNamespace(
                enable_sql_result_limit=True,
                sql_result_limit=5,
            )
            result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        mock_result = mock_connection.execute.return_value
        executed_sql = str(mock_connection.execute.call_args.args[0])

        assert "LIMIT 5" in executed_sql
        mock_result.fetchmany.assert_called_once_with(5)
        mock_result.fetchall.assert_not_called()
        assert "limited to first 5 rows" in result["messages"][0].content

    def test_execute_sql_node_can_disable_result_limit(self, mock_llm, mock_catalog):
        """Test SQL execution can opt out of the configured row limit."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM users")

        with patch("openchatbi.text2sql.generate_sql.config") as mock_config:
            mock_config.get.return_value = SimpleNamespace(
                enable_sql_result_limit=False,
                sql_result_limit=5,
            )
            result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        mock_result = mock_connection.execute.return_value
        executed_sql = str(mock_connection.execute.call_args.args[0])

        assert "openchatbi_limited_result" not in executed_sql
        assert "LIMIT 5" not in executed_sql
        mock_result.fetchall.assert_called_once()
        mock_result.fetchmany.assert_not_called()
        assert "limited to first" not in result["messages"][0].content

    def test_execute_sql_node_limit_disabled_invalid_prefix_returns_syntax_error(self, mock_llm, mock_catalog):
        """Test malformed SQL prefix returns syntax error when limit wrapper is disabled."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="<SQL> SELECT * FROM users")

        with patch("openchatbi.text2sql.generate_sql.config") as mock_config:
            mock_config.get.return_value = SimpleNamespace(
                enable_sql_result_limit=False,
                sql_result_limit=5,
            )
            mock_engine = mock_catalog.get_sql_engine.return_value
            mock_connection = mock_engine.connect.return_value.__enter__.return_value
            from sqlalchemy.exc import ProgrammingError

            mock_connection.execute.side_effect = ProgrammingError("", "", 'line 1:1: mismatched input "<"')
            result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR, SQL_SYNTAX_ERROR

        assert result["sql_execution_result"] == SQL_SYNTAX_ERROR
        assert result["sql_execution_result"] != SQL_SECURITY_ERROR
        assert "previous_sql_errors" in result
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL syntax error"

    def test_execute_sql_node_allows_with_select_query(self, mock_llm, mock_catalog):
        """Test SQL execution allows CTE-based SELECT queries."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(
            messages=[],
            sql="WITH active_users AS (SELECT * FROM users WHERE active = 1) SELECT * FROM active_users",
        )

        result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        executed_sql = str(mock_connection.execute.call_args.args[0])

        assert result["sql_execution_result"] == "SQL_SUCCESS"
        assert "WITH active_users AS" in executed_sql

    def test_execute_sql_node_rejects_disallowed_operation_after_select(self, mock_llm, mock_catalog):
        """Test SQL execution rejects disallowed operations after a SELECT."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM users; DELETE FROM users")

        result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == SQL_SECURITY_ERROR
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL security error"
        assert "Operation not allowed" in result["previous_sql_errors"][-1]["error"]
        mock_connection.execute.assert_not_called()

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM users",
            "DELETE users FROM users",
            "UPDATE users SET name = 'Jane'",
            "INSERT INTO users VALUES (1, 'Jane')",
            "INSERT OVERWRITE TABLE users SELECT * FROM archived_users",
            "DROP TABLE users",
            "DROP DATABASE analytics",
            "DROP MATERIALIZED VIEW user_summary",
            "CREATE TABLE users (id INT)",
            "CREATE OR REPLACE VIEW user_summary AS SELECT * FROM users",
            "ALTER TABLE users ADD COLUMN email TEXT",
            "ALTER USER readonly_user SET password = 'secret'",
            "TRUNCATE TABLE users",
            "SELECT * FROM users INTO OUTFILE '/tmp/users.csv'",
            "SELECT * FROM users FOR UPDATE",
        ],
    )
    def test_execute_sql_node_rejects_disallowed_operations(self, mock_llm, mock_catalog, sql):
        """Test SQL execution rejects write and DDL operations."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql=sql)

        result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == SQL_SECURITY_ERROR
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL security error"
        mock_connection.execute.assert_not_called()

    def test_execute_sql_node_allows_ddl_words_in_string_literals(self, mock_llm, mock_catalog):
        """Test operation words in string literals do not trigger object-level checks."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(
            messages=[],
            sql="SELECT * FROM audit_logs WHERE action IN ('INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER')",
        )

        result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == "SQL_SUCCESS"
        mock_connection.execute.assert_called_once()

    def test_execute_sql_node_allows_select_into_table(self, mock_llm, mock_catalog):
        """Test SELECT INTO table syntax is not blocked by the file export rule."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * INTO archived_users FROM users")

        result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == "SQL_SUCCESS"
        mock_connection.execute.assert_called_once()

    def test_execute_sql_node_allows_operation_phrase_after_quote(self, mock_llm, mock_catalog):
        """Test operation phrases after quotes are not treated as SQL operations."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM audit_logs WHERE action='ALTER TABLE'")

        result = execute_node(state)

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == "SQL_SUCCESS"
        mock_connection.execute.assert_called_once()

    def test_execute_sql_node_rejects_disallowed_operations_in_comments(self, mock_llm, mock_catalog):
        """Test dangerous keywords in comments are rejected by the safety check."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="/* harmless */ SELECT * FROM users -- ignored DROP TABLE users")

        result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == SQL_SECURITY_ERROR
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL security error"
        mock_connection.execute.assert_not_called()

    def test_execute_sql_node_rejects_dangerous_statement_after_comment(self, mock_llm, mock_catalog):
        """Test dangerous statements remain blocked when comments are present."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM users /* comment */; DROP TABLE users")

        result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value

        assert result["sql_execution_result"] == SQL_SECURITY_ERROR
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL security error"
        mock_connection.execute.assert_not_called()

    def test_execute_sql_node_empty_sql(self, mock_llm, mock_catalog):
        """Test SQL execution with empty SQL."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(messages=[], sql="")  # Empty SQL

        result = execute_node(state)

        assert "sql_execution_result" in result
        from openchatbi.constants import SQL_NA

        assert result["sql_execution_result"] == SQL_NA

    def test_execute_sql_node_syntax_error(self, mock_llm, mock_catalog):
        """Test SQL execution with syntax error."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

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
        _, _, regenerate_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

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
            mock_retriever.invoke.return_value = []

            result = regenerate_node(state)

        assert "sql" in result
        assert "sql_retry_count" in result
        assert result["sql_retry_count"] == 2

    def test_should_generate_visualization_or_retry_success(self):
        """Test routing to visualization with successful execution."""
        state = SQLGraphState(sql_execution_result=SQL_SUCCESS, sql_retry_count=1)

        result = _should_generate_visualization_or_retry(state)
        assert result == "generate_visualization"

    def test_should_generate_visualization_or_retry_timeout(self):
        """Test routing ends on database timeout."""
        state = SQLGraphState(sql_execution_result=SQL_EXECUTE_TIMEOUT, sql_retry_count=1)

        result = _should_generate_visualization_or_retry(state)
        assert result == "end"

    def test_should_generate_visualization_or_retry_retry_needed(self):
        """Test routing to SQL regeneration when retry is needed."""
        state = SQLGraphState(sql_execution_result="SYNTAX_ERROR", sql_retry_count=1)

        result = _should_generate_visualization_or_retry(state)
        assert result == "regenerate_sql"

    def test_should_generate_visualization_or_retry_max_retries_reached(self):
        """Test routing ends when max retries are reached."""
        state = SQLGraphState(sql_execution_result="SYNTAX_ERROR", sql_retry_count=3)

        result = _should_generate_visualization_or_retry(state)
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

    def test_should_execute_sql_with_null_sql(self):
        """Test execute decision treats NULL as no SQL."""
        state = SQLGraphState(sql="NULL")

        result = should_execute_sql(state)
        assert result == "end"

    def test_sql_generation_with_examples(self, mock_llm, mock_catalog):
        """Test SQL generation with relevant examples."""
        generate_node, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

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
            mock_retriever.invoke.return_value = [mock_document]

            with patch(
                "openchatbi.text2sql.generate_sql.sql_example_dicts",
                {"How many users are there?": ("SELECT COUNT(*) FROM users", ["users"])},
            ):
                result = generate_node(state)

        assert "sql" in result

    def test_sql_error_handling_database_error(self, mock_llm, mock_catalog):
        """Test handling of database connection errors."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

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

    def test_sql_error_handling_operational_syntax_error(self, mock_llm, mock_catalog):
        """Test handling of syntax errors wrapped by OperationalError."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import OperationalError

        mock_connection.execute.side_effect = OperationalError(
            "",
            {},
            Exception('near "<": syntax error'),
        )

        state = SQLGraphState(messages=[], sql="SELECT * FROM users WHERE < invalid")
        result = execute_node(state)

        from openchatbi.constants import SQL_SYNTAX_ERROR

        assert result["sql_execution_result"] == SQL_SYNTAX_ERROR
        assert "previous_sql_errors" in result
        assert result["previous_sql_errors"][-1]["error_type"] == "SQL syntax error"

    def test_sql_error_handling_operational_timeout_takes_priority(self, mock_llm, mock_catalog):
        """Test timeout markers take precedence over syntax markers."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import OperationalError

        mock_connection.execute.side_effect = OperationalError(
            "",
            {},
            Exception('connection timed out near "<": syntax error'),
        )

        state = SQLGraphState(messages=[], sql="SELECT * FROM users WHERE < invalid")
        result = execute_node(state)

        from openchatbi.constants import SQL_EXECUTE_TIMEOUT

        assert result["sql_execution_result"] == SQL_EXECUTE_TIMEOUT

    def test_sql_error_handling_operational_unknown_error(self, mock_llm, mock_catalog):
        """Test non-timeout, non-syntax operational errors are treated as unknown."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import OperationalError

        mock_connection.execute.side_effect = OperationalError(
            "",
            {},
            Exception("disk i/o error"),
        )

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")
        result = execute_node(state)

        from openchatbi.constants import SQL_UNKNOWN_ERROR

        assert result["sql_execution_result"] == SQL_UNKNOWN_ERROR
        assert "previous_sql_errors" in result
        assert result["previous_sql_errors"][-1]["error_type"] == "Database operational error"

    def test_regenerate_sql_empty_response(self, mock_llm, mock_catalog):
        """Test regeneration with empty LLM response."""
        mock_llm.invoke.return_value = AIMessage(content="")

        _, _, regenerate_node, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

        state = SQLGraphState(
            messages=[],
            question="Show all users",
            rewrite_question="Show all users",
            tables=[{"table": "users", "columns": []}],
            previous_sql_errors=[],
            sql_retry_count=1,
        )

        with patch("openchatbi.text2sql.generate_sql.sql_example_retriever") as mock_retriever:
            mock_retriever.invoke.return_value = []

            result = regenerate_node(state)

        assert "sql" in result
        assert result["sql"] == ""
        from openchatbi.constants import SQL_NA

        assert result["sql_execution_result"] == SQL_NA

    # --- Task 9: enriched previous_sql_errors fields + empty-result default OFF ---

    def test_execute_sql_node_syntax_error_enriched_fields(self, mock_llm, mock_catalog):
        """Syntax errors carry new structured fields without changing legacy strings."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import ProgrammingError

        mock_connection.execute.side_effect = ProgrammingError("", "", "Syntax error")
        state = SQLGraphState(messages=[], sql="SELECT * FRON users")

        result = execute_node(state)

        from openchatbi.constants import SQL_SYNTAX_ERROR

        entry = result["previous_sql_errors"][-1]
        # Legacy human-readable contract preserved:
        assert entry["error_type"] == "SQL syntax error"
        assert entry["error"].startswith("SQL syntax error:")
        # New structured fields:
        assert entry["error_code"] == SQL_SYNTAX_ERROR
        assert entry["error_class"] == "SQLSyntaxError"
        assert entry["recovery_strategy"] == "retry"
        assert entry["attempt"] == 1

    def test_execute_sql_node_security_error_enriched_fields(self, mock_llm, mock_catalog):
        """Security errors keep legacy strings and gain surface_to_user strategy."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        state = SQLGraphState(messages=[], sql="SELECT * FROM users; DELETE FROM users")

        result = execute_node(state)

        from openchatbi.constants import SQL_SECURITY_ERROR

        entry = result["previous_sql_errors"][-1]
        assert entry["error_type"] == "SQL security error"
        assert entry["error_code"] == SQL_SECURITY_ERROR
        assert entry["error_class"] == "SQLSecurityError"
        assert entry["recovery_strategy"] == "surface_to_user"

    def test_execute_sql_node_attempt_increments_with_history(self, mock_llm, mock_catalog):
        """attempt counts existing previous_sql_errors + 1."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        from sqlalchemy.exc import ProgrammingError

        mock_connection.execute.side_effect = ProgrammingError("", "", "Syntax error")
        state = SQLGraphState(
            messages=[],
            sql="SELECT * FRON users",
            previous_sql_errors=[
                {"sql": "x", "error": "SQL syntax error: x", "error_type": "SQL syntax error"}
            ],
        )

        result = execute_node(state)
        assert result["previous_sql_errors"][-1]["attempt"] == 2

    def test_execute_sql_node_empty_result_default_success(self, mock_llm, mock_catalog):
        """Zero-row results stay SQL_SUCCESS when the empty-result gate is off (default)."""
        _, execute_node, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
        mock_engine = mock_catalog.get_sql_engine.return_value
        mock_connection = mock_engine.connect.return_value.__enter__.return_value
        mock_result = mock_connection.execute.return_value
        mock_result.fetchmany.return_value = []
        mock_result.fetchall.return_value = []

        state = SQLGraphState(messages=[], sql="SELECT * FROM users")
        result = execute_node(state)

        assert result["sql_execution_result"] == SQL_SUCCESS
