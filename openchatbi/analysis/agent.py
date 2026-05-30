"""Data Analysis Deep Agent implementation."""

import logging
from typing import Callable

from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from deepagents import create_deep_agent
from openchatbi.llm.llm import get_llm
from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.timeseries_forecast import check_forecast_service_health, timeseries_forecast

from openchatbi.tool.anomaly_detection import anomaly_detection
from openchatbi.tool.adtributor_tool import adtributor_drilldown

logger = logging.getLogger(__name__)


def _load_data_analysis_prompt() -> str:
    """Load the data analysis prompt template."""
    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "data_analysis_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Data analysis prompt file not found at {prompt_path}")
        return "You are a data analysis agent. Help the user analyze their data."


def build_data_analysis_agent(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the data analysis deep agent.

    Args:
        sql_graph: Compiled SQL generation graph to use for text2sql tool.
        sync_mode: Whether to use synchronous mode.
        llm_provider: LLM provider to use.
        checkpointer: Checkpointer for state persistence.
        memory_store: Store for long-term memory.

    Returns:
        CompiledStateGraph: The compiled data analysis deep agent.
    """
    # Import here to avoid circular imports
    from openchatbi.agent_graph import get_sql_tools
    
    # 1. Prepare tools
    text2sql_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)
    
    tools = [text2sql_tool, run_python_code]
    
    # Add forecast and anomaly detection if service is healthy
    if check_forecast_service_health():
        tools.append(timeseries_forecast)
        tools.append(anomaly_detection)
    else:
        logger.warning("Time series forecasting service is not healthy. Skipping forecast and anomaly detection tools.")
        
    tools.append(adtributor_drilldown)

    # 2. Get LLM
    llm = get_llm(llm_provider)
    
    # 3. Get prompt
    system_prompt = _load_data_analysis_prompt()

    # 4. Create Deep Agent
    # Note: deepagents.create_deep_agent expects a model string or a BaseChatModel
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=memory_store,
    )
    
    return agent


class DataAnalysisInput(BaseModel):
    """Input schema for data analysis tool."""
    reasoning: str = Field(description="Reason for delegating to the data analysis agent")
    task: str = Field(description="Full description of the analysis task, including metrics, dimensions, time range, and analysis type")


def get_data_analysis_tool(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> StructuredTool:
    """Create the data analysis tool that delegates to the deep agent.

    Args:
        sql_graph: Compiled SQL generation graph.
        sync_mode: Whether to use synchronous mode.
        llm_provider: LLM provider to use.
        checkpointer: Checkpointer for state persistence.
        memory_store: Store for long-term memory.

    Returns:
        StructuredTool: The data analysis tool.
    """
    agent = build_data_analysis_agent(
        sql_graph=sql_graph,
        sync_mode=sync_mode,
        llm_provider=llm_provider,
        checkpointer=checkpointer,
        memory_store=memory_store,
    )

    def call_data_analysis_sync(reasoning: str, task: str) -> str:
        """Sync function for data analysis tool."""
        logger.info(f"Delegating to data analysis agent (sync). Reasoning: {reasoning}, Task: {task}")
        from langgraph.errors import GraphInterrupt
        import traceback
        try:
            # Deep Agents expect input in a specific format, usually {"messages": [HumanMessage(content=task)]}
            # or just {"messages": task}
            response = agent.invoke({"messages": task})
            # Extract the final message content
            if "messages" in response and response["messages"]:
                return response["messages"][-1].content
            return str(response)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return f"Error occurred during data analysis: {str(e)}"

    async def call_data_analysis_async(reasoning: str, task: str) -> str:
        """Async function for data analysis tool."""
        logger.info(f"Delegating to data analysis agent (async). Reasoning: {reasoning}, Task: {task}")
        from langgraph.errors import GraphInterrupt
        import traceback
        try:
            response = await agent.ainvoke({"messages": task})
            if "messages" in response and response["messages"]:
                return response["messages"][-1].content
            return str(response)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return f"Error occurred during data analysis: {str(e)}"

    description = """Delegate complex data analysis tasks to a specialized sub-agent.
Use this tool for:
1. Single metric trend forecasting
2. Single metric anomaly detection
3. Single metric anomaly drill-down (root cause analysis)
4. Multi-metric correlation
5. Business combination analysis

Provide a detailed task description including metrics, dimensions, time ranges, and the specific analysis goal."""

    if sync_mode:
        return StructuredTool.from_function(
            func=call_data_analysis_sync,
            name="data_analysis",
            description=description,
            args_schema=DataAnalysisInput,
            return_direct=False,
        )
    else:
        return StructuredTool.from_function(
            coroutine=call_data_analysis_async,
            name="data_analysis",
            description=description,
            args_schema=DataAnalysisInput,
            return_direct=False,
        )
