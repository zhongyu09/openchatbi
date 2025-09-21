"""State classes for OpenChatBI graph execution."""

from typing import Any

from langgraph.types import Send
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the main agent graph execution.

    Extends MessagesState with additional fields for routing and responses.
    """

    agent_next_node: str
    sends: list[Send]
    sql: str
    final_answer: str


class SQLGraphState(MessagesState):
    """State for SQL generation subgraph.

    Contains rewritten question, table selection, extracted entities, and generated SQL.
    """

    rewrite_question: str
    tables: list[dict[str, Any]]
    info_entities: dict[str, Any]
    sql: str
    sql_retry_count: int
    sql_execution_result: str
    schema_info: dict[str, Any]  # Data schema analysis results
    data: str  # CSV data for display
    previous_sql_errors: list[dict[str, Any]]
    visualization_dsl: dict[str, Any]


class InputState(MessagesState):
    """Input state schema for the main graph."""

    pass


class OutputState(MessagesState):
    """Output state schema for the main graph."""

    pass


class SQLOutputState(MessagesState):
    """Output state schema for the SQL generation subgraph."""

    rewrite_question: str
    tables: list[dict[str, Any]]
    sql: str
    schema_info: dict[str, Any]  # Data schema analysis results
    data: str  # CSV data for display
    visualization_dsl: dict[str, Any]
