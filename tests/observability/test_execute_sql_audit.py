"""execute_sql_node emits a masked SQL audit record via AuditLogger.

Drives execution via the same mock_catalog.get_sql_engine pattern used in
tests/test_text2sql_generate_sql.py (error-path tests). Uses the real
create_sql_nodes(llm, catalog, dialect) signature and unpacks the returned
6-tuple to get execute_sql_node at position 1.
"""

import logging
from unittest.mock import MagicMock, Mock

from openchatbi.graph_state import SQLGraphState
from openchatbi.observability.context import set_run_context
from openchatbi.text2sql.generate_sql import create_sql_nodes


def _make_mock_catalog(rows=None):
    """Build a mock catalog with a functioning SQL engine context manager."""
    if rows is None:
        rows = [("1",)]
    catalog = Mock()
    catalog.get_table_information.return_value = {
        "description": "test table",
        "sql_rule": "",
        "derived_metric": "",
    }
    catalog.get_column_list.return_value = []

    mock_engine = Mock()
    mock_connection = Mock()
    mock_result = Mock()
    mock_result.fetchall.return_value = rows
    mock_result.fetchmany.return_value = rows
    mock_result.keys.return_value = ["count"]
    mock_connection.execute.return_value = mock_result

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_connection
    mock_context_manager.__exit__.return_value = None
    mock_engine.connect.return_value = mock_context_manager

    catalog.get_sql_engine.return_value = mock_engine
    return catalog


def test_execute_sql_node_audits_success(caplog) -> None:
    """execute_sql_node must emit a masked openchatbi.audit record on success."""
    set_run_context("alice", "req-1")

    mock_llm = Mock()
    mock_catalog = _make_mock_catalog(rows=[("42",)])

    # create_sql_nodes(llm, catalog, dialect) — real 6-tuple signature
    _, execute_sql_node, _, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")

    state = SQLGraphState(messages=[], sql="SELECT COUNT(*) FROM users WHERE id = 7")

    with caplog.at_level(logging.INFO, logger="openchatbi.audit"):
        result = execute_sql_node(state)

    assert result["sql_execution_result"] == "SQL_SUCCESS"

    audit_lines = [r.message for r in caplog.records if r.name == "openchatbi.audit"]
    assert audit_lines, "No openchatbi.audit record emitted"
    assert any("sql_exec" in m and "SQL_SUCCESS" in m for m in audit_lines)
    # Numeric literal '7' must be masked — the raw "= 7" must not appear
    assert all("= 7" not in m for m in audit_lines)


def test_execute_sql_node_audits_operational_timeout(caplog) -> None:
    """execute_sql_node emits an audit record with SQL_CHECK_TIMEOUT on timeout."""
    set_run_context("bob", "req-2")

    mock_llm = Mock()
    mock_catalog = _make_mock_catalog()

    mock_connection = mock_catalog.get_sql_engine.return_value.connect.return_value.__enter__.return_value
    from sqlalchemy.exc import OperationalError

    mock_connection.execute.side_effect = OperationalError(
        "", {}, Exception("connection timed out")
    )

    _, execute_sql_node, _, _, _, _ = create_sql_nodes(mock_llm, mock_catalog, "presto")
    state = SQLGraphState(messages=[], sql="SELECT * FROM users")

    with caplog.at_level(logging.INFO, logger="openchatbi.audit"):
        result = execute_sql_node(state)

    from openchatbi.constants import SQL_EXECUTE_TIMEOUT

    assert result["sql_execution_result"] == SQL_EXECUTE_TIMEOUT

    audit_lines = [r.message for r in caplog.records if r.name == "openchatbi.audit"]
    assert audit_lines, "No openchatbi.audit record emitted on timeout"
    # SQL_EXECUTE_TIMEOUT value is "SQL_CHECK_TIMEOUT"
    assert any("sql_exec" in m for m in audit_lines)
