"""Constants used throughout the OpenChatBI application."""

# Date/time format strings
datetime_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"
datetime_format_ms = "%Y-%m-%d %H:%M:%S.%f"
datetime_format_ms_T = "%Y-%m-%dT%H:%M:%S.%fZ"

# SQL execution status codes
SQL_NA = "SQL_NA"
SQL_SUCCESS = "SQL_SUCCESS"
SQL_EXECUTE_TIMEOUT = "SQL_CHECK_TIMEOUT"
SQL_SYNTAX_ERROR = "SQL_SYNTAX_ERROR"
SQL_UNKNOWN_ERROR = "SQL_UNKNOWN_ERROR"


MCP_TOOL_DEFAULT_TIMEOUT_SECONDS = 60
