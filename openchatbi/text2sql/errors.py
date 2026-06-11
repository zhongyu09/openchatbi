"""Structured error taxonomy and classifier for the Text2SQL subgraph.

Keeps the SQL_* status codes and the human-readable error_type strings that
downstream tests are coupled to; new structured information (recovery strategy,
error class, error code) is carried alongside without mutating those strings.
"""

from enum import Enum

from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError, TimeoutError

from openchatbi.constants import (
    SQL_EXECUTE_TIMEOUT,
    SQL_NA,
    SQL_SECURITY_ERROR,
    SQL_SYNTAX_ERROR,
    SQL_UNKNOWN_ERROR,
)


class RecoveryStrategy(str, Enum):
    """How the graph should react to a classified Text2SQL error."""

    RETRY = "retry"
    RETRY_WITH_NEW_TABLE = "retry_with_new_table"
    SURFACE_TO_USER = "surface_to_user"
    ABORT = "abort"


class Text2SQLError(Exception):
    """Base structured error for the Text2SQL subgraph.

    Attributes:
        code: One of the existing SQL_* status code constants (downstream-compatible).
        recovery_strategy: How the graph should recover (see RecoveryStrategy).
        error_type: Human-readable label preserved for legacy test/UI coupling.
        user_message: Optional message safe to surface to the end user.
        orig: The originating exception, if any.
    """

    code: str = SQL_UNKNOWN_ERROR
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    error_type: str = "Unexpected error"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
        error_type: str | None = None,
        user_message: str | None = None,
        orig: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if recovery_strategy is not None:
            self.recovery_strategy = recovery_strategy
        if error_type is not None:
            self.error_type = error_type
        self.user_message = user_message
        self.orig = orig


class SQLSecurityError(Text2SQLError):
    """Raised when generated SQL fails safety validation."""

    code = SQL_SECURITY_ERROR
    recovery_strategy = RecoveryStrategy.SURFACE_TO_USER
    error_type = "SQL security error"


class SQLSyntaxError(Text2SQLError):
    """Raised when the database reports a SQL syntax/parse error."""

    code = SQL_SYNTAX_ERROR
    recovery_strategy = RecoveryStrategy.RETRY
    error_type = "SQL syntax error"


class InvalidDBConnectionError(Text2SQLError):
    """Raised when the database connection itself is invalid/unauthorized."""

    code = SQL_EXECUTE_TIMEOUT
    recovery_strategy = RecoveryStrategy.SURFACE_TO_USER
    error_type = "Database connection error"


class DBTimeoutError(Text2SQLError):
    """Raised on database timeout / dropped connection during execution."""

    code = SQL_EXECUTE_TIMEOUT
    recovery_strategy = RecoveryStrategy.ABORT
    error_type = "Database connection timeout"


class EmptyResultError(Text2SQLError):
    """Raised (opt-in only) when a query returns zero rows."""

    code = SQL_NA  # 见约定#6:软失败码,非 SQL_SUCCESS;默认 gate 关闭时根本不构造此异常
    recovery_strategy = RecoveryStrategy.RETRY_WITH_NEW_TABLE
    error_type = "Empty result"


class UnknownSQLError(Text2SQLError):
    """Catch-all for operational/database/unexpected errors."""

    code = SQL_UNKNOWN_ERROR
    recovery_strategy = RecoveryStrategy.RETRY
    error_type = "Unexpected error"


def classify_sql_exception(exc: BaseException) -> Text2SQLError:
    """Classify a raw exception into a structured Text2SQLError.

    Reuses the existing message extraction and operational-error heuristics so
    classification stays consistent with the legacy multi-branch except chain.
    The returned error's ``error_type`` matches the exact human-readable strings
    that downstream tests assert on.
    """
    from openchatbi.text2sql.generate_sql import (
        _classify_operational_error,
        _extract_exception_message,
    )

    if isinstance(exc, Text2SQLError):
        return exc

    if isinstance(exc, TimeoutError):
        return DBTimeoutError(_extract_exception_message(exc), orig=exc)

    if isinstance(exc, OperationalError):
        category = _classify_operational_error(exc)
        if category == "timeout_or_connection":
            return DBTimeoutError(str(exc), orig=exc)
        if category == "syntax":
            return SQLSyntaxError(str(exc), orig=exc)
        # Non-timeout, non-syntax operational error -> preserve legacy label/code.
        return UnknownSQLError(str(exc), code=SQL_UNKNOWN_ERROR, error_type="Database operational error", orig=exc)

    if isinstance(exc, ProgrammingError):
        return SQLSyntaxError(str(exc), orig=exc)

    if isinstance(exc, DatabaseError):
        return UnknownSQLError(str(exc), code=SQL_UNKNOWN_ERROR, error_type="Database error", orig=exc)

    return UnknownSQLError(str(exc), error_type="Unexpected error", orig=exc)
