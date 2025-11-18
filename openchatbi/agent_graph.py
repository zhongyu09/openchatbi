"""Main agent graph construction and execution logic."""

import datetime
import logging
import traceback
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send, interrupt
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.catalog import CatalogStore
from openchatbi.constants import datetime_format
from openchatbi.context_config import get_context_config
from openchatbi.context_manager import ContextManager
from openchatbi.graph_state import AgentState, InputState, OutputState
from openchatbi.llm.llm import call_llm_chat_model_with_retry, get_default_llm
from openchatbi.prompts.system_prompt import get_agent_prompt_template
from openchatbi.text2sql.sql_graph import build_sql_graph
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.mcp_tools import create_mcp_tools_sync, get_mcp_tools_async
from openchatbi.tool.memory import get_memory_tools
from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.save_report import save_report
from openchatbi.tool.search_knowledge import search_knowledge, show_schema
from openchatbi.tool.timeseries_forecast import check_forecast_service_health, timeseries_forecast
from openchatbi.utils import log, recover_incomplete_tool_calls

logger = logging.getLogger(__name__)


def get_mcp_servers():
    """Get MCP servers from config with fallback for tests."""
    try:
        return config.get().mcp_servers
    except ValueError:
        return []


def ask_human(state: AgentState) -> dict[str, Any]:
    """Node function to ask human for additional information or clarification.

    Args:
        state (AgentState): The current graph state containing messages and context.

    Returns:
        dict: Updated state with human feedback as a tool message and user input.
    """
    tool_call = state["messages"][-1].tool_calls[0]
    tool_call_id = tool_call["id"]
    args = tool_call["args"]
    user_feedback = interrupt({"text": args["question"], "buttons": args.get("options", None)})
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": user_feedback}]
    return {
        "messages": tool_message,
        "history_messages": [AIMessage(args["question"]), HumanMessage(user_feedback)],
        "user_input": user_feedback,
    }


class CallSQLGraphInput(BaseModel):
    reasoning: str = Field(
        description="Explanation of why Text2SQL tool is needed",
    )
    context: str = Field(
        description="""The full context pass to Text2SQL tool, make sure do not miss any potential information that related to user's question.
        Following the format: History Conversation: (user and assistant history dialog)
        Information: (the knowledge you retrival that is relevant, like metrics and dimensions)
        User's latest question:""",
    )


# Description for SQL tools
TEXT2SQL_TOOL_DESCRIPTION = """Text2SQL tool to generate and execute SQL query and build visualization DSL for UI
based on user's question and context.

Returns:
    str: A formatted response containing SQL, data, and visualization status.

Important notes:
- If user want to change the visualization chart type or style, add the requirement in the question
- Make sure to provide question in English
"""


def _format_sql_response(sql_graph_response: dict) -> str:
    """Format SQL graph response into a standardized string format.

    Args:
        sql_graph_response: The response dictionary from the SQL graph

    Returns:
        str: Formatted response string
    """
    sql = sql_graph_response.get("sql", "")
    data = sql_graph_response.get("data", "")
    visualization_dsl = sql_graph_response.get("visualization_dsl", {})

    response_parts = []
    if sql:
        response_parts.append(f"SQL Query:\n```sql\n{sql}\n```")
    if data:
        response_parts.append(f"\nQuery Results (CSV format):\n```csv\n{data}\n```")

    # Include visualization status
    if visualization_dsl and "error" not in visualization_dsl:
        chart_type = visualization_dsl.get("chart_type", "unknown")
        response_parts.append(
            f"\nVisualization Created: {chart_type} chart has been automatically generated and will be displayed in the UI."
        )
    elif visualization_dsl and "error" in visualization_dsl:
        response_parts.append(f"\nVisualization Error: {visualization_dsl['error']}")

    return "\n\n".join(response_parts) if response_parts else "No results returned."


def get_sql_tools(sql_graph: CompiledStateGraph, sync_mode: bool = False) -> Callable:
    """Create SQL generation tool from compiled SQL graph.

    Args:
        sql_graph (CompiledStateGraph): The compiled SQL generation subgraph.
        sync_mode (bool): Whether to create synchronous or asynchronous tools

    Returns:
        function: Tool function for SQL generation.
    """

    def call_sql_graph_sync(reasoning: str, context: str) -> str:
        """Sync node function for Text2SQL tool"""
        log(f"Call SQL graph (sync) with reasoning: {reasoning}, context: {context}")
        try:
            sql_graph_response = sql_graph.invoke({"messages": context})
            return _format_sql_response(sql_graph_response)
        except GraphInterrupt as e:
            log(f"Sql graph interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"Run sql graph error:\n{repr(e)}")
            traceback.print_exc()
        return "Error occurred when calling Text2SQL tool."

    async def call_sql_graph_async(reasoning: str, context: str) -> str:
        """Async node function for Text2SQL tool"""
        log(f"Call SQL graph (async) with reasoning: {reasoning}, context: {context}")
        try:
            sql_graph_response = await sql_graph.ainvoke({"messages": context})
            return _format_sql_response(sql_graph_response)
        except GraphInterrupt as e:
            log(f"Sql graph interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"Run sql graph error:\n{repr(e)}")
            traceback.print_exc()
        return "Error occurred when calling Text2SQL tool."

    if sync_mode:
        return StructuredTool.from_function(
            func=call_sql_graph_sync,
            name="text2sql",
            description=TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            return_direct=False,
        )
    else:
        return StructuredTool.from_function(
            coroutine=call_sql_graph_async,
            name="text2sql",
            description=TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            return_direct=False,
        )


def agent_llm_call(llm: BaseChatModel, tools: list, context_manager: ContextManager = None) -> Callable:
    """Create llm call function to generate reasoning and determine next node based on tool calls in LLM response.

    Args:
        llm (BaseChatModel): The LLM for agent decision-making.
        tools: List of tools.
        context_manager: Optional context manager for handling long conversations.

    Returns:
        function: function that processes state and determines next node.
    """

    # OpenAI models support strict tool calling
    if isinstance(llm, BaseChatOpenAI):
        llm_with_tools = llm.bind_tools(tools, strict=True)
    else:
        llm_with_tools = llm.bind_tools(tools)

    def _call_model(state: AgentState):
        # First, check and recover any incomplete tool calls
        recovery_ops = recover_incomplete_tool_calls(state)
        if recovery_ops:
            return {"messages": recovery_ops, "agent_next_node": "llm_node"}

        messages = state["messages"]
        final_messages = []
        if isinstance(messages[-1], HumanMessage):
            final_messages.append(messages[-1])

        # Apply context management if available (before processing)
        if context_manager:
            original_count = len(messages)
            context_manager.manage_context_messages(messages)
            if len(messages) != original_count:
                logger.info(f"Context management: modified messages from {original_count} to {len(messages)}")

        system_prompt = get_agent_prompt_template().replace(
            "[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format)
        )

        response = call_llm_chat_model_with_retry(
            llm_with_tools,
            ([SystemMessage(system_prompt)] + messages),
            streaming_tokens=True,
            bound_tools=tools,
            parallel_tool_call=True,
        )
        if isinstance(response, AIMessage):
            tool_calls = response.tool_calls
            print("Tool Call:", ", ".join(tool["name"] for tool in tool_calls))
            if tool_calls:
                # Group tool calls by type for parallel routing
                ask_human_calls = [call for call in tool_calls if call["name"] == "AskHuman"]
                normal_tool_calls = [call for call in tool_calls if call["name"] != "AskHuman"]

                # Create Send objects for parallel routing
                sends = []
                if ask_human_calls:
                    # Create message with only AskHuman calls
                    ask_human_msg = AIMessage(content=response.content, tool_calls=ask_human_calls)
                    sends.append(Send("ask_human", {"messages": [ask_human_msg]}))

                if normal_tool_calls:
                    # Create message with only normal tool calls
                    tool_msg = AIMessage(content=response.content, tool_calls=normal_tool_calls)
                    sends.append(Send("use_tool", {"messages": [tool_msg]}))

                return {"messages": [response], "history_messages": final_messages, "sends": sends}
            else:
                final_messages.append(AIMessage(response.content))
                return {
                    "messages": [response],
                    "final_answer": response.content,
                    "history_messages": final_messages,
                    "agent_next_node": END,
                }
        elif response is None:
            return {
                "messages": [AIMessage("Sorry, the LLM service is currently unavailable.")],
                "history_messages": final_messages,
                "agent_next_node": END,
            }
        else:
            return {"messages": [response], "history_messages": final_messages, "agent_next_node": END}

    return _call_model


def _build_graph_core(
    catalog: CatalogStore,
    sync_mode: bool,
    checkpointer: Checkpointer,
    memory_store: BaseStore,
    memory_tools: tuple[Callable, Callable] | None,
    mcp_tools: list,
    enable_context_management: bool = True,
) -> CompiledStateGraph:
    """Core graph building logic shared by both sync and async versions.

    Args:
        catalog: Catalog store containing schema information
        sync_mode: Whether to use synchronous mode for tools and operations
        checkpointer: The Checkpointer for state persistence
        memory_store: The BaseStore to use for long-term memory
        memory_tools: Tuple of (manage_memory_tool, search_memory_tool)
        mcp_tools: Pre-initialized MCP tools
        enable_context_management: Whether to enable context management

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution
    """
    sql_graph = build_sql_graph(catalog, checkpointer, memory_store)
    call_sql_graph_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    # Use provided memory tools or create them
    if memory_tools:
        manage_memory_tool, search_memory_tool = memory_tools
    else:
        manage_memory_tool, search_memory_tool = get_memory_tools(
            get_default_llm(), sync_mode=sync_mode, store=memory_store
        )

    log(str(mcp_tools))
    normal_tools = [
        search_knowledge,
        show_schema,
        call_sql_graph_tool,
        run_python_code,
        manage_memory_tool,
        search_memory_tool,
        save_report,
    ]
    if check_forecast_service_health():
        normal_tools.append(timeseries_forecast)
    else:
        logger.warning("Time series forecasting service is not healthy. Skipping timeseries_forecast tool.")
    normal_tools.extend(mcp_tools)

    # Initialize context manager if enabled
    context_manager = None
    if enable_context_management:
        context_manager = ContextManager(llm=get_default_llm(), config=get_context_config())

    tool_node = ToolNode(normal_tools)

    # Define the agent graph
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    # Add nodes to the graph
    graph.add_node("llm_node", agent_llm_call(get_default_llm(), normal_tools + [AskHuman], context_manager))
    graph.add_node("ask_human", ask_human)
    graph.add_node("use_tool", tool_node)

    # Add edges between nodes
    graph.add_edge(START, "llm_node")
    graph.add_edge("ask_human", "llm_node")
    graph.add_edge("use_tool", "llm_node")

    # Add conditional routing from llm node
    def route_tools(state: AgentState):
        # Only use sends if the last message came from the llm node (has tool_calls)
        last_message = state["messages"][-1] if state["messages"] else None
        if (
            last_message
            and isinstance(last_message, AIMessage)
            and last_message.tool_calls
            and "sends" in state
            and state["sends"]
        ):
            return state["sends"]  # Return Send objects for parallel execution
        elif "agent_next_node" in state:
            return state["agent_next_node"]  # Return single node name
        else:
            return END

    graph.add_conditional_edges(
        "llm_node",
        route_tools,
        # mapping of paths to node names (for single routing)
        {
            "llm_node": "llm_node",
            "ask_human": "ask_human",
            "use_tool": "use_tool",
            END: END,
        },
    )

    graph = graph.compile(name="agent_graph", checkpointer=checkpointer, store=memory_store)
    return graph


def build_agent_graph_sync(
    catalog: CatalogStore,
    checkpointer: Checkpointer = None,
    memory_store: BaseStore = None,
    enable_context_management: bool = True,
) -> CompiledStateGraph:
    """Build the main agent graph with all nodes and edges (sync version).

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        enable_context_management: Whether to enable context management for long conversations.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for sync context
    mcp_tools = create_mcp_tools_sync(get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=True,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=None,  # Always None for sync version - creates its own
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
    )


async def build_agent_graph_async(
    catalog: CatalogStore,
    checkpointer: Checkpointer = None,
    memory_store: BaseStore = None,
    memory_tools: tuple[Callable, Callable] = None,
    enable_context_management: bool = True,
) -> CompiledStateGraph:
    """Build the main agent graph with all nodes and edges (async version).

    This function is identical to build_agent_graph_sync but properly handles
    async MCP tool initialization when called from async contexts.

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        memory_tools: Tuple of (manage_memory_tool, search_memory_tool). If None, creates async tools.
        enable_context_management: Whether to enable context management for long conversations.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for async context
    mcp_tools = await get_mcp_tools_async(get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=False,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=memory_tools,
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
    )
