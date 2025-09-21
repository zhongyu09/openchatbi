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
from openchatbi.constants import SQL_SUCCESS
from openchatbi.graph_state import InputState, SQLGraphState, SQLOutputState
from openchatbi.llm.llm import default_llm, text2sql_llm
from openchatbi.text2sql.extraction import information_extraction, information_extraction_conditional_edges
from openchatbi.text2sql.generate_sql import create_sql_nodes, should_execute_sql, should_retry_sql
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


def should_generate_visualization_or_retry(state: SQLGraphState) -> str:
    """Conditional edge function to determine next action after execute_sql.

    Args:
        state (SQLGraphState): Current state

    Returns:
        str: Next node name - "generate_visualization" if SQL succeeded, "regenerate_sql" if retry needed, "end" if done
    """
    execution_result = state.get("sql_execution_result", "")
    retry_count = state.get("sql_retry_count", 0)
    max_retries = 3

    if execution_result == SQL_SUCCESS:
        return "generate_visualization"
    elif retry_count < max_retries and execution_result not in ("SQL_EXECUTE_TIMEOUT",):
        return "regenerate_sql"
    else:
        return "end"


def build_sql_graph(catalog: CatalogStore, checkpointer: Checkpointer, memory_store: BaseStore) -> CompiledStateGraph:
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
    if isinstance(default_llm, BaseChatOpenAI):
        llm_with_tools = default_llm.bind_tools(tools, strict=True).bind(response_format={"type": "json_object"})
    else:
        llm_with_tools = default_llm.bind_tools(tools)
    # Create SQL processing nodes with visualization configuration
    generate_sql_node, execute_sql_node, regenerate_sql_node, generate_visualization_node = create_sql_nodes(
        text2sql_llm, catalog, dialect=config.get().dialect, visualization_mode=config.get().visualization_mode
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

    # Add conditional edges for execute_sql - either retry, generate visualization, or end
    graph.add_conditional_edges(
        "execute_sql",
        should_generate_visualization_or_retry,
        {
            "generate_visualization": "generate_visualization",
            "regenerate_sql": "regenerate_sql",
            "end": END,
        },
    )

    # Add edge from visualization to end
    graph.add_edge("generate_visualization", END)

    graph = graph.compile(name="text2sql_graph", checkpointer=checkpointer, store=memory_store)
    return graph
