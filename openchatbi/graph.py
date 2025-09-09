"""Main agent graph construction and execution logic."""

import datetime
import logging
import traceback
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, interrupt
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.constants import datetime_format
from openchatbi.graph_state import AgentState, InputState, OutputState
from openchatbi.llm.llm import call_llm_chat_model_with_retry, default_llm
from openchatbi.prompts.system_prompt import AGENT_PROMPT_TEMPLATE
from openchatbi.text2sql.sql_graph import build_sql_graph
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.memory import get_memory_tools
from openchatbi.tool.run_python_code import run_python_code
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


def get_sql_tools(sql_graph: CompiledStateGraph, sync_mode: bool = False) -> Callable:
    """Create SQL generation tool from compiled SQL graph.

    Args:
        sql_graph (CompiledStateGraph): The compiled SQL generation subgraph.
        sync_mode (bool): Whether to create synchronous or asynchronous tools

    Returns:
        function: Tool function for SQL generation.
    """

    if sync_mode:

        @tool("text2sql", args_schema=CallSQLGraphInput, return_direct=False, infer_schema=True)
        def call_sql_graph_sync(reasoning: str, context: str) -> (str, str):
            """Text2SQL tool (sync version) to generate and execute SQL query based on user's question and context."""
            log(f"Call SQL graph (sync) with reasoning: {reasoning}, context: {context}")
            try:
                sql_graph_response = sql_graph.invoke({"messages": context})
                sql = sql_graph_response.get("sql")
                data = sql_graph_response.get("data")
                return sql, data
            except Exception as e:
                log(f"Run sql graph error:\n{repr(e)}")
                traceback.print_exc()
            return "Error occurred when calling Text2SQL tool.", ""

        return call_sql_graph_sync
    else:

        @tool("text2sql", args_schema=CallSQLGraphInput, return_direct=False, infer_schema=True)
        async def call_sql_graph_async(reasoning: str, context: str) -> (str, str):
            """Text2SQL tool (async version) to generate and execute SQL query based on user's question and context.
            Returns:
                sql (str): The generated SQL query.
                data (str): The CSV data returned from executing the SQL query.
            """
            log(f"Call SQL graph (async) with reasoning: {reasoning}, context: {context}")
            try:
                sql_graph_response = await sql_graph.ainvoke({"messages": context})
                sql = sql_graph_response.get("sql")
                data = sql_graph_response.get("data")
                return sql, data
            except Exception as e:
                log(f"Run sql graph error:\n{repr(e)}")
                traceback.print_exc()
            return "Error occurred when calling Text2SQL tool.", ""

        return call_sql_graph_async


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
            "[basic_knowledge_glossary]", config.get().bi_config.get("basic_knowledge_glossary", "")
        ).replace("[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format))

        response = call_llm_chat_model_with_retry(
            llm_with_tools, ([SystemMessage(system_prompt)] + messages), bound_tools=tools
        )
        agent_next_node = ""
        if isinstance(response, AIMessage):
            tool_calls = response.tool_calls
            if tool_calls:
                if tool_calls[0]["name"] == "AskHuman":
                    agent_next_node = "ask_human"
                elif tool_calls[0]["name"] in (
                    "search_knowledge",
                    "show_schema",
                    "text2sql",
                    "run_python_code",
                    "manage_memory",
                    "search_memory",
                ):
                    agent_next_node = "use_tool"
                else:
                    raise ValueError(f"Unknown tool call: {tool_calls[0]['name']}")
            else:
                return {"messages": [response], "final_answer": response.content, "agent_next_node": END}
        return {"messages": [response], "agent_next_node": agent_next_node}

    return _call_model


def build_agent_graph(
    catalog: Any,
    sync_mode: bool = False,
    checkpointer: Checkpointer = None,
    memory_store: BaseStore = None,
    memory_tools: tuple[Callable, Callable] = None,
) -> CompiledStateGraph:
    """Build the main agent graph with all nodes and edges.

    Args:
        catalog: Catalog store containing schema information.
        sync_mode: Whether to use synchronous mode for tools and operations.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        memory_tools: Tuple of (manage_memory_tool, search_memory_tool). If None, creates the sync tools if sync_mode = True.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    sql_graph = build_sql_graph(catalog, checkpointer, memory_store)
    call_sql_graph_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    # Use provided memory tools or create them
    if memory_tools:
        manage_memory_tool, search_memory_tool = memory_tools
    else:
        manage_memory_tool, search_memory_tool = get_memory_tools(default_llm, sync_mode=sync_mode, store=memory_store)

    normal_tools = [
        search_knowledge,
        show_schema,
        call_sql_graph_tool,
        run_python_code,
        manage_memory_tool,
        search_memory_tool,
    ]
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
    graph.add_conditional_edges(
        "router",
        lambda state: state["agent_next_node"],
        # mapping of paths to node names
        {
            "ask_human": "ask_human",
            "use_tool": "use_tool",
            END: END,
        },
    )

    graph = graph.compile(name="agent_graph", checkpointer=checkpointer, store=memory_store)
    return graph
