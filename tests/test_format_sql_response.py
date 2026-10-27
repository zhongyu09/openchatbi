"""Unit tests for `_format_sql_response` in `openchatbi.agent_graph`."""

import pytest
from langchain_core.messages import AIMessage

from openchatbi.agent_graph import _format_sql_response
from openchatbi.constants import SQL_EXECUTE_TIMEOUT, SQL_SECURITY_ERROR, SQL_SUCCESS


class TestFormatSqlResponse:
    """Cover the propagation of SQL sub-graph labels and errors."""

    def test_normal_result_without_limit(self):
        response = {
            "sql": "SELECT * FROM t",
            "data": "a,b\n1,2",
            "schema_info": {"row_count": 1},
            "sql_execution_result": SQL_SUCCESS,
        }
        result = _format_sql_response(response)
        assert "SQL Query:" in result
        assert "Query Results (CSV format):" in result
        assert "limited to" not in result.lower()

    def test_truncation_notice_when_limit_reached(self):
        response = {
            "sql": "SELECT * FROM t",
            "data": "a\n" + "\n".join(str(i) for i in range(10000)),
            "schema_info": {"row_count": 10000, "result_limit": 10000},
            "sql_execution_result": SQL_SUCCESS,
        }
        result = _format_sql_response(response)
        assert "limited to first 10000 rows" in result
        assert "may contain more rows" in result

    def test_no_truncation_notice_when_below_limit(self):
        response = {
            "sql": "SELECT * FROM t",
            "data": "a\n1\n2\n3",
            "schema_info": {"row_count": 3, "result_limit": 10000},
            "sql_execution_result": SQL_SUCCESS,
        }
        result = _format_sql_response(response)
        assert "limited to" not in result.lower()

    def test_timeout_error_forwarded(self):
        error_msg = "```sql\nSELECT 1\n```\nDatabase Connection Timeout: timed out\nPlease check database connectivity."
        response = {
            "sql": "SELECT 1",
            "data": "",
            "sql_execution_result": SQL_EXECUTE_TIMEOUT,
            "messages": [AIMessage(error_msg)],
        }
        result = _format_sql_response(response)
        assert "Database Connection Timeout" in result
        assert "Please check database connectivity" in result

    def test_security_error_forwarded(self):
        error_msg = "```sql\nSELECT 1\n```\nSQL security error: unsafe operation"
        response = {
            "sql": "SELECT 1",
            "data": "",
            "sql_execution_result": SQL_SECURITY_ERROR,
            "messages": [AIMessage(error_msg)],
        }
        result = _format_sql_response(response)
        assert "SQL security error" in result

    def test_empty_response(self):
        result = _format_sql_response({})
        assert result == "No results returned."

    @pytest.mark.parametrize(
        "row_count",
        [None, "not-an-int", 10000.0],
    )
    def test_truncation_notice_skipped_for_non_integer_row_count(self, row_count):
        response = {
            "sql": "SELECT * FROM t",
            "data": "a\n1",
            "schema_info": {"row_count": row_count, "result_limit": 10000},
            "sql_execution_result": SQL_SUCCESS,
        }
        result = _format_sql_response(response)
        assert "limited to" not in result.lower()
