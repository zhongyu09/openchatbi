"""Tests for the structured Text2SQL error taxonomy and classifier."""

from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError, TimeoutError

from openchatbi.constants import (
    SQL_EXECUTE_TIMEOUT,
    SQL_NA,
    SQL_SECURITY_ERROR,
    SQL_SYNTAX_ERROR,
    SQL_UNKNOWN_ERROR,
)
from openchatbi.text2sql.errors import (
    DBTimeoutError,
    EmptyResultError,
    InvalidDBConnectionError,
    RecoveryStrategy,
    SQLSecurityError,
    SQLSyntaxError,
    Text2SQLError,
    UnknownSQLError,
    classify_sql_exception,
)


class TestRecoveryStrategy:
    def test_enum_values(self):
        assert RecoveryStrategy.RETRY == "retry"
        assert RecoveryStrategy.RETRY_WITH_NEW_TABLE == "retry_with_new_table"
        assert RecoveryStrategy.SURFACE_TO_USER == "surface_to_user"
        assert RecoveryStrategy.ABORT == "abort"

    def test_is_str_enum(self):
        assert isinstance(RecoveryStrategy.RETRY, str)


class TestText2SQLErrorSubclasses:
    def test_base_fields(self):
        orig = ValueError("boom")
        err = Text2SQLError(
            "msg",
            code=SQL_UNKNOWN_ERROR,
            recovery_strategy=RecoveryStrategy.RETRY,
            user_message="please retry",
            orig=orig,
        )
        assert err.code == SQL_UNKNOWN_ERROR
        assert err.recovery_strategy is RecoveryStrategy.RETRY
        assert err.user_message == "please retry"
        assert err.orig is orig

    def test_security_error_defaults(self):
        err = SQLSecurityError("Operation not allowed")
        assert err.code == SQL_SECURITY_ERROR
        assert err.recovery_strategy is RecoveryStrategy.SURFACE_TO_USER
        assert isinstance(err, Text2SQLError)

    def test_syntax_error_defaults(self):
        err = SQLSyntaxError("bad syntax")
        assert err.code == SQL_SYNTAX_ERROR
        assert err.recovery_strategy is RecoveryStrategy.RETRY

    def test_invalid_connection_defaults(self):
        err = InvalidDBConnectionError("bad creds")
        assert err.code == SQL_EXECUTE_TIMEOUT
        assert err.recovery_strategy is RecoveryStrategy.SURFACE_TO_USER

    def test_db_timeout_defaults(self):
        err = DBTimeoutError("timed out")
        assert err.code == SQL_EXECUTE_TIMEOUT
        assert err.recovery_strategy is RecoveryStrategy.ABORT

    def test_empty_result_defaults(self):
        err = EmptyResultError("no rows")
        assert err.code == SQL_NA  # 见约定#6:EmptyResultError.code = SQL_NA
        assert err.recovery_strategy is RecoveryStrategy.RETRY_WITH_NEW_TABLE

    def test_unknown_error_defaults(self):
        err = UnknownSQLError("disk i/o error")
        assert err.code == SQL_UNKNOWN_ERROR
        assert err.recovery_strategy is RecoveryStrategy.RETRY


class TestClassifySqlException:
    def test_security_error_passthrough(self):
        out = classify_sql_exception(SQLSecurityError("Operation not allowed: x"))
        assert isinstance(out, SQLSecurityError)
        assert out.code == SQL_SECURITY_ERROR
        assert out.error_type == "SQL security error"

    def test_timeout_error(self):
        out = classify_sql_exception(TimeoutError("query timed out"))
        assert isinstance(out, DBTimeoutError)
        assert out.code == SQL_EXECUTE_TIMEOUT
        assert out.error_type == "Database connection timeout"

    def test_operational_timeout_or_connection(self):
        out = classify_sql_exception(OperationalError("", {}, Exception("connection refused")))
        assert isinstance(out, DBTimeoutError)
        assert out.code == SQL_EXECUTE_TIMEOUT

    def test_operational_syntax(self):
        out = classify_sql_exception(OperationalError("", {}, Exception('near "<": syntax error')))
        assert isinstance(out, SQLSyntaxError)
        assert out.code == SQL_SYNTAX_ERROR
        assert out.error_type == "SQL syntax error"

    def test_operational_other_is_operational(self):
        out = classify_sql_exception(OperationalError("", {}, Exception("disk i/o error")))
        assert out.code == SQL_UNKNOWN_ERROR
        assert out.error_type == "Database operational error"

    def test_programming_error_is_syntax(self):
        out = classify_sql_exception(ProgrammingError("", "", "Syntax error"))
        assert isinstance(out, SQLSyntaxError)
        assert out.code == SQL_SYNTAX_ERROR
        assert out.error_type == "SQL syntax error"

    def test_database_error_is_unknown(self):
        out = classify_sql_exception(DatabaseError("", "", "generic db error"))
        assert out.code == SQL_UNKNOWN_ERROR
        assert out.error_type == "Database error"

    def test_generic_exception_is_unknown(self):
        out = classify_sql_exception(RuntimeError("something else"))
        assert isinstance(out, UnknownSQLError)
        assert out.code == SQL_UNKNOWN_ERROR
        assert out.error_type == "Unexpected error"

    def test_orig_is_preserved(self):
        src = ProgrammingError("", "", "Syntax error")
        out = classify_sql_exception(src)
        assert out.orig is src
