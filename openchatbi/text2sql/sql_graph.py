"""SQL generation graph construction and execution."""

from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, interrupt

from openchatbi import config
from openchatbi.catalog import CatalogStore
from openchatbi.constants import SQL_EXECUTE_TIMEOUT, SQL_SUCCESS
from openchatbi.text2sql.errors import RecoveryStrategy
from openchatbi.graph_state import InputState, SQLGraphState, SQLOutputState
from openchatbi.llm.llm import get_llm, get_text2sql_llm
from openchatbi.text2sql.data import get_learned_sql_store
from openchatbi.text2sql.extraction import information_extraction, information_extraction_conditional_edges
from openchatbi.text2sql.generate_sql import create_sql_nodes, should_execute_sql
from openchatbi.text2sql.schema_linking import schema_linking
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.search_knowledge import search_knowledge


def ask_human(state):
    """Node function to ask human for additional information or clarification.

    Args:
        state (SQLGraphState): The current SQL graph state containing messages and context.

    Returns:
        dict: Updated state with human feedback as a tool message and user input.
    """
    tool_call = state["messages"][-1].tool_calls[0]
    tool_call_id = tool_call["id"]
    args = tool_call["args"]
    user_feedback = interrupt({"text": args["question"], "buttons": args.get("options", None)})
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": user_feedback}]
    return {"messages": tool_message, "user_input": user_feedback}


def _get_sql_retry_config() -> tuple[int, bool]:
    """Read retry settings from Config, defaulting to legacy values."""
    try:
        cfg = config.get()
    except ValueError:
        return 3, False
    max_retries = getattr(cfg, "sql_max_retries", 3)
    if not isinstance(max_retries, int) or max_retries < 0:
        max_retries = 3
    return max_retries, bool(getattr(cfg, "retry_on_timeout", False))


def _should_generate_visualization_or_retry(state: SQLGraphState) -> str:
    """Conditional edge function to determine next action after execute_sql.

    Routing is strategy-driven: the last classified error's recovery_strategy
    decides whether to regenerate or end. Falls back to legacy code-based routing
    when no recovery_strategy is present (e.g. timeouts, untouched states).

    Args:
        state (SQLGraphState): Current state

    Returns:
        str: "generate_visualization" on success, "regenerate_sql" to retry, "end" otherwise.
    """
    execution_result = state.get("sql_execution_result", "")
    retry_count = state.get("sql_retry_count", 0)
    max_retries, retry_on_timeout = _get_sql_retry_config()

    if execution_result == SQL_SUCCESS:
        return "generate_visualization"

    # Timeouts are classified with non-retry strategies (ABORT / SURFACE_TO_USER),
    # which would make retry_on_timeout dead config if left to strategy routing;
    # honor the explicit opt-in before the strategy-driven branch.
    if execution_result == SQL_EXECUTE_TIMEOUT:
        if retry_on_timeout and retry_count < max_retries:
            return "regenerate_sql"
        return "end"

    previous_errors = state.get("previous_sql_errors", [])
    strategy = previous_errors[-1].get("recovery_strategy") if previous_errors else None

    if strategy is not None:
        if strategy in (RecoveryStrategy.SURFACE_TO_USER.value, RecoveryStrategy.ABORT.value):
            return "end"
        if strategy in (RecoveryStrategy.RETRY.value, RecoveryStrategy.RETRY_WITH_NEW_TABLE.value):
            return "regenerate_sql" if retry_count < max_retries else "end"
        return "end"

    # Legacy fallback: no structured strategy recorded.
    if retry_count < max_retries and (
        execution_result != SQL_EXECUTE_TIMEOUT or retry_on_timeout
    ):
        return "regenerate_sql"
    return "end"


def route_after_confidence(state: SQLGraphState) -> str:
    """Route after the confidence gate based on the human decision.

    approve -> visualization; reject -> regenerate; edit -> re-execute the
    user-edited SQL. Defaults to visualization when no decision is present
    (gate disabled or score above threshold).
    """
    decision = state.get("human_sql_decision", "approve")
    if decision == "reject":
        return "regenerate_sql"
    if decision == "edit":
        return "execute_sql"
    return "generate_visualization"


def build_sql_graph(
    catalog: CatalogStore, checkpointer: Checkpointer, memory_store: BaseStore, llm_provider: str | None = None
) -> CompiledStateGraph:
    """Build SQL generation graph with all nodes and edges.

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, no long-term memory.

    Returns:
        CompiledStateGraph: Compiled SQL graph ready for execution.
    """
    tools = [search_knowledge, AskHuman]
    search_tool_node = ToolNode([search_knowledge])
    default_llm = get_llm(llm_provider)
    if isinstance(default_llm, BaseChatOpenAI):
        llm_with_tools = default_llm.bind_tools(tools, strict=True).bind(response_format={"type": "json_object"})
    else:
        llm_with_tools = default_llm.bind_tools(tools)
    # Create SQL processing nodes with visualization configuration
    (
        generate_sql_node,
        execute_sql_node,
        regenerate_sql_node,
        generate_visualization_node,
        score_sql_node,
        confidence_gate_node,
    ) = create_sql_nodes(
        get_text2sql_llm(llm_provider),
        catalog,
        dialect=config.get().dialect,
        visualization_mode=config.get().visualization_mode,
        learned_sql_store=get_learned_sql_store(),
    )

    # Define the SQL generation graph
    graph = StateGraph(SQLGraphState, input_schema=InputState, output_schema=SQLOutputState)

    # Add nodes to the graph
    graph.add_node("search_knowledge", search_tool_node)
    graph.add_node("ask_human", ask_human)
    graph.add_node("information_extraction", information_extraction(llm_with_tools))
    graph.add_node("table_selection", schema_linking(default_llm, catalog))
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("regenerate_sql", regenerate_sql_node)
    graph.add_node("generate_visualization", generate_visualization_node)
    graph.add_node("score_sql", score_sql_node)
    graph.add_node("confidence_gate", confidence_gate_node)

    # Add basic edges
    graph.add_edge(START, "information_extraction")
    graph.add_edge("ask_human", "information_extraction")
    graph.add_edge("search_knowledge", "information_extraction")
    graph.add_edge("table_selection", "generate_sql")

    # Add conditional routing from information extraction
    graph.add_conditional_edges(
        "information_extraction",
        information_extraction_conditional_edges,
        # mapping of paths to node names
        {
            "ask_human": "ask_human",
            "search_knowledge": "search_knowledge",
            "next": "table_selection",
            "end": END,
        },
    )

    # Add conditional edges for generate_sql
    graph.add_conditional_edges(
        "generate_sql",
        should_execute_sql,
        {
            "execute_sql": "execute_sql",
            "end": END,
        },
    )

    # Add conditional edges for regenerate_sql
    graph.add_conditional_edges(
        "regenerate_sql",
        should_execute_sql,
        {
            "execute_sql": "execute_sql",
            "end": END,
        },
    )

    # Add conditional edges for execute_sql - either retry, score the SQL
    # (post_exec confidence gate), or end. On success we route to score_sql,
    # which feeds the confidence gate before visualization.
    graph.add_conditional_edges(
        "execute_sql",
        _should_generate_visualization_or_retry,
        {
            "generate_visualization": "score_sql",
            "regenerate_sql": "regenerate_sql",
            "end": END,
        },
    )

    # score_sql -> confidence_gate -> {visualization | regenerate | re-execute}.
    # When the gate is disabled (default), confidence_gate returns "approve"
    # immediately and route_after_confidence sends control to visualization,
    # preserving the prior SUCCESS -> visualization behavior.
    graph.add_edge("score_sql", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_confidence,
        {
            "generate_visualization": "generate_visualization",
            "regenerate_sql": "regenerate_sql",
            "execute_sql": "execute_sql",
        },
    )

    # Add edge from visualization to end
    graph.add_edge("generate_visualization", END)

    graph = graph.compile(name="text2sql_graph", checkpointer=checkpointer, store=memory_store)
    return graph
