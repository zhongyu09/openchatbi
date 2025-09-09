"""State classes for OpenChatBI graph execution."""

from typing import Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the main agent graph execution.

    Extends MessagesState with additional fields for routing and responses.
    """

    agent_next_node: str
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
    data: str
    previous_sql_errors: list[dict[str, Any]]


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
    data: str
