"""Main agent graph construction and execution logic."""

import datetime
import logging
import traceback
from collections.abc import Callable
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, interrupt, Send
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.catalog import CatalogStore
from openchatbi.constants import datetime_format
from openchatbi.graph_state import AgentState, InputState, OutputState
from openchatbi.llm.llm import call_llm_chat_model_with_retry, default_llm
from openchatbi.prompts.system_prompt import AGENT_PROMPT_TEMPLATE
from openchatbi.text2sql.sql_graph import build_sql_graph
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.mcp_tools import create_mcp_tools_sync, get_mcp_tools_async
from openchatbi.tool.memory import get_memory_tools
from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.save_report import save_report
from openchatbi.tool.search_knowledge import search_knowledge, show_schema
from openchatbi.utils import log

logger = logging.getLogger(__name__)


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
    return {"messages": tool_message, "user_input": user_feedback}


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


def agent_router(llm: BaseChatModel, tools: list) -> Callable:
    """Create router function to determine next node based on LLM tool calls.

    Args:
        llm (BaseChatModel): The LLM for decision-making.
        tools: List of tools.

    Returns:
        function: Router function that processes state and determines next node.
    """

    # OpenAI models support strict tool calling
    if isinstance(llm, BaseChatOpenAI):
        llm_with_tools = llm.bind_tools(tools, strict=True)
    else:
        llm_with_tools = llm.bind_tools(tools)

    def _call_model(state: AgentState):
        messages = state["messages"]
        system_prompt = AGENT_PROMPT_TEMPLATE.replace(
            "[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format)
        )

        response = call_llm_chat_model_with_retry(
            llm_with_tools, ([SystemMessage(system_prompt)] + messages), bound_tools=tools, parallel_tool_call=True
        )
        agent_next_node = ""
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

                return {"messages": [response], "sends": sends}
            else:
                return {"messages": [response], "final_answer": response.content, "agent_next_node": END}
        elif response is None:
            return {"messages": [AIMessage("Sorry, the LLM service is currently unavailable.")], "agent_next_node": END}
        else:
            return {"messages": [response], "agent_next_node": END}

    return _call_model


def _build_graph_core(
    catalog: CatalogStore,
    sync_mode: bool,
    checkpointer: Checkpointer,
    memory_store: BaseStore,
    memory_tools: Optional[tuple[Callable, Callable]],
    mcp_tools: list,
) -> CompiledStateGraph:
    """Core graph building logic shared by both sync and async versions.

    Args:
        catalog: Catalog store containing schema information
        sync_mode: Whether to use synchronous mode for tools and operations
        checkpointer: The Checkpointer for state persistence
        memory_store: The BaseStore to use for long-term memory
        memory_tools: Tuple of (manage_memory_tool, search_memory_tool)
        mcp_tools: Pre-initialized MCP tools

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution
    """
    sql_graph = build_sql_graph(catalog, checkpointer, memory_store)
    call_sql_graph_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    # Use provided memory tools or create them
    if memory_tools:
        manage_memory_tool, search_memory_tool = memory_tools
    else:
        manage_memory_tool, search_memory_tool = get_memory_tools(default_llm, sync_mode=sync_mode, store=memory_store)

    log(str(mcp_tools))
    normal_tools = [
        search_knowledge,
        show_schema,
        call_sql_graph_tool,
        run_python_code,
        manage_memory_tool,
        search_memory_tool,
        save_report,
    ] + mcp_tools
    tool_node = ToolNode(normal_tools)

    # Define the agent graph
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    # Add nodes to the graph
    graph.add_node("router", agent_router(default_llm, normal_tools + [AskHuman]))
    graph.add_node("ask_human", ask_human)
    graph.add_node("use_tool", tool_node)

    # Add edges between nodes
    graph.add_edge(START, "router")
    graph.add_edge("ask_human", "router")
    graph.add_edge("use_tool", "router")

    # Add conditional routing from router node
    def route_tools(state: AgentState):
        # Only use sends if the last message came from the router (has tool_calls)
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
        "router",
        route_tools,
        # mapping of paths to node names (for single routing)
        {
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
) -> CompiledStateGraph:
    """Build the main agent graph with all nodes and edges (sync version).

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for sync context
    mcp_tools = create_mcp_tools_sync(config.get().mcp_servers)

    return _build_graph_core(
        catalog=catalog,
        sync_mode=True,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=None,  # Always None for sync version - creates its own
        mcp_tools=mcp_tools,
    )


async def build_agent_graph_async(
    catalog: CatalogStore,
    checkpointer: Checkpointer = None,
    memory_store: BaseStore = None,
    memory_tools: tuple[Callable, Callable] = None,
) -> CompiledStateGraph:
    """Build the main agent graph with all nodes and edges (async version).

    This function is identical to build_agent_graph_sync but properly handles
    async MCP tool initialization when called from async contexts.

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        memory_tools: Tuple of (manage_memory_tool, search_memory_tool). If None, creates async tools.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for async context
    mcp_tools = await get_mcp_tools_async(config.get().mcp_servers)

    return _build_graph_core(
        catalog=catalog,
        sync_mode=False,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=memory_tools,
        mcp_tools=mcp_tools,
    )
