"""Data Analysis Agent implementation."""

import logging
from typing import cast

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from openchatbi.constants import DEFAULT_ANALYSIS_RECURSION_LIMIT
from openchatbi.llm.llm import get_analysis_llm
from openchatbi.tool.adtributor_tool import adtributor_drilldown
from openchatbi.tool.anomaly_detection import anomaly_detection
from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.search_knowledge import search_schema, show_schema
from openchatbi.tool.timeseries_forecast import check_forecast_service_health, timeseries_forecast

logger = logging.getLogger(__name__)

# Markers used to surface data_analysis sub-agent failures to the parent agent.
DATA_ANALYSIS_ERROR_PREFIX = "Error occurred during data analysis:"
DATA_ANALYSIS_FAILED_MARKER = "[DATA_ANALYSIS_FAILED]"


def _is_transient_error(error: Exception) -> bool:
    """Heuristic: classify network/rate-limit/timeout errors as transient.

    Walks the exception chain (``__cause__`` / ``__context__``) so that a
    low-level transient error wrapped in a generic carrier exception (e.g. an
    ``httpx.HTTPStatusError`` wrapping a ``ReadTimeout``, or an
    ``openai.APIStatusError`` carrying a 429/503) is still detected.
    """
    transient_keywords = {
        "rate",
        "limit",
        "timeout",
        "connection",
        "connect",
        "network",
        "serviceunavailable",
        "internalserver",
        "temporarily",
    }
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        if any(kw in name for kw in transient_keywords):
            return True
        # Some HTTP client exceptions expose the status code; 429 (rate limit)
        # and 5xx (server errors) are conventionally retryable.
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
        current = current.__cause__ or current.__context__
    return False


def _classify_data_analysis_error(error: Exception) -> tuple[str, str]:
    """Classify a sub-agent failure as retryable or permanent.

    Returns:
        A tuple of (tag, user_message). ``tag`` is ``[RETRYABLE]`` or ``[PERMANENT]``.
    """
    # HITL must bubble up to the parent graph; never swallow it.
    if isinstance(error, GraphInterrupt):
        raise error

    if isinstance(error, GraphRecursionError):
        return "[RETRYABLE]", (
            "The analysis required more steps than allowed. "
            "Try narrowing the time range, reducing dimensions, or splitting into smaller sub-tasks."
        )

    if _is_transient_error(error):
        return "[RETRYABLE]", (
            "A transient error occurred while running the analysis. "
            "You may retry once; if it still fails, report the issue to the user."
        )

    # Everything else is treated as permanent (implementation bug, bad args,
    # auth failure, unrecoverable service error, etc.).
    return "[PERMANENT]", str(error)


def _format_data_analysis_error(error: Exception) -> str:
    """Format a sub-agent error into a structured string the parent can parse."""
    tag, message = _classify_data_analysis_error(error)
    return f"{DATA_ANALYSIS_FAILED_MARKER}{tag} {DATA_ANALYSIS_ERROR_PREFIX} {message}"


def _build_sub_agent_config(config: RunnableConfig | None) -> RunnableConfig:
    """Derive an isolated child config for the data analysis sub-agent.

    The sub-agent is a separate compiled graph that may share the parent's
    checkpointer. To avoid clobbering the parent's checkpoint thread (and to
    provide the ``thread_id`` LangGraph requires when a checkpointer is set),
    we derive a deterministic child ``thread_id`` from the parent one.

    We also pin a fixed, single-level ``checkpoint_ns`` (``"data_analysis"``)
    instead of popping it. ``checkpoint_ns`` doubles as the subgraph namespace
    for ``astream(subgraphs=True)``: popping it would flatten the sub-agent's
    events onto the parent's depth-0 stream (making its steps indistinguishable
    from the main agent's in a streaming UI), whereas a fixed single-level
    namespace surfaces the sub-agent as a proper nested subgraph (depth >= 1).
    Checkpoint isolation is provided by the derived ``thread_id`` plus the
    ``checkpoint_id`` reset below, not by ``checkpoint_ns``. ``callbacks`` and
    ``tags`` are propagated as-is. ``recursion_limit`` and ``metadata`` are
    intentionally *not* propagated: tools see ``recursion_limit=25`` from
    LangChain's default config, which would override the sub-agent's intended
    limit, and parent metadata belongs to the main trace. Instead,
    ``recursion_limit`` is set explicitly from OpenChatBI configuration.
    """
    sub_config: RunnableConfig = cast(RunnableConfig, {})
    configurable: dict = {}
    if config:
        # Drop ``configurable`` (handled separately), ``recursion_limit`` and
        # ``metadata`` so the sub-agent uses its own bound/explicit limits and
        # does not inherit the parent LangGraph/LangChain defaults.
        sub_config = cast(
            RunnableConfig,
            {k: v for k, v in config.items() if k not in ("configurable", "metadata", "recursion_limit")},
        )
        configurable = dict(config.get("configurable") or {})

    parent_thread_id = configurable.get("thread_id")
    if parent_thread_id is not None:
        configurable["thread_id"] = f"{parent_thread_id}:data_analysis"
        # Pin a clean, single-level namespace so the sub-agent streams as a
        # nested subgraph rather than being flattened onto the parent stream.
        configurable["checkpoint_ns"] = "data_analysis"
    # Resetting checkpoint_id is what actually prevents the sub-agent from
    # resuming off the parent's checkpoint (the real isolation guarantee).
    configurable.pop("checkpoint_id", None)

    sub_config["configurable"] = configurable
    # Always set an explicit recursion limit so the sub-agent never accidentally
    # inherits the parent/tool-node default of 25.
    sub_config["recursion_limit"] = _resolve_analysis_recursion_limit()
    return sub_config


def _extract_final_content(response: dict) -> str:
    """Extract the final message content from the agent response as a string."""
    if isinstance(response, dict) and response.get("messages"):
        content = response["messages"][-1].content
        if isinstance(content, str):
            return content
        # Content may be a list of content blocks (e.g. multimodal); join text parts.
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", "") or "")
            return "\n".join(p for p in parts if p)
        return str(content)
    return str(response)


def _resolve_analysis_recursion_limit() -> int:
    """Resolve the data analysis sub-agent recursion limit from application config.

    Falls back to the package default when config is unavailable (e.g. in tests).
    """
    from openchatbi.config_loader import ConfigLoader

    try:
        return ConfigLoader().get().analysis_recursion_limit
    except ValueError:
        return DEFAULT_ANALYSIS_RECURSION_LIMIT


def _resolve_min_input_length() -> int:
    """Resolve the forecasting model's minimum input length for prompt injection.

    Falls back to the default when config/service are unavailable (e.g. in tests).
    """
    from openchatbi import config
    from openchatbi.tool.timeseries_forecast import DEFAULT_MIN_FORECAST_INPUT_LENGTH, get_min_forecast_input_length

    try:
        service_url = config.get().timeseries_forecasting_service_url
        return get_min_forecast_input_length(service_url)
    except Exception:
        return DEFAULT_MIN_FORECAST_INPUT_LENGTH


def _load_data_analysis_prompt() -> str:
    """Load the data analysis prompt template, injecting the model's minimum input length."""
    import os

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "data_analysis_prompt.md")
    try:
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logger.warning(f"Data analysis prompt file not found at {prompt_path}")
        return "You are a data analysis agent. Help the user analyze their data."

    return template.replace("[min_input_length]", str(_resolve_min_input_length()))


def build_data_analysis_agent(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build the data analysis agent.

    Args:
        sql_graph: Compiled SQL generation graph to use for text2sql tool.
        sync_mode: Whether to use synchronous mode.
        llm_provider: LLM provider to use.
        checkpointer: Checkpointer for state persistence.
        memory_store: Store for long-term memory.

    Returns:
        CompiledStateGraph: The compiled data analysis agent.
    """
    # Import here to avoid circular imports
    from openchatbi.agent_graph import get_sql_tools

    # 1. Prepare tools
    text2sql_tool = get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode)

    tools = [search_schema, show_schema, text2sql_tool, run_python_code]

    # Add forecast and anomaly detection if service is healthy
    if check_forecast_service_health():
        tools.append(timeseries_forecast)
        tools.append(anomaly_detection)
    else:
        logger.warning("Time series forecasting service is not healthy. Skipping forecast and anomaly detection tools.")

    tools.append(adtributor_drilldown)

    # 2. Get LLM (dedicated analysis_llm if configured, else default)
    llm = get_analysis_llm(llm_provider)

    # 3. Get prompt
    system_prompt = _load_data_analysis_prompt()

    # 4. Create the agent (built on the deepagents framework)
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
    task: str = Field(
        description=(
            "Full, self-contained description of the analysis task. The data analysis agent does NOT "
            "know the current time or the conversation context, so the task MUST be unambiguous: "
            "resolve every relative time expression (e.g. 'last week', 'today', 'yesterday') into "
            "CONCRETE absolute dates/datetimes, and explicitly state the metric(s), dimensions, "
            "granularity, time range, and analysis type. For anomaly detection, also state the "
            "evaluation window (the period of interest) and the end date of the analysis window."
        )
    )


def get_data_analysis_tool(
    sql_graph: CompiledStateGraph,
    sync_mode: bool = False,
    llm_provider: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    memory_store: BaseStore | None = None,
) -> StructuredTool:
    """Create the data analysis tool that delegates to the data analysis agent.

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

    def call_data_analysis_sync(reasoning: str, task: str, config: RunnableConfig) -> str:
        """Sync function for data analysis tool."""
        logger.info(f"Delegating to data analysis agent (sync). Reasoning: {reasoning}, Task: {task}")
        import traceback

        sub_config = _build_sub_agent_config(config)
        try:
            # The agent accepts a string under "messages", which is coerced to a HumanMessage.
            response = agent.invoke({"messages": task}, config=sub_config)
            return _extract_final_content(response)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return _format_data_analysis_error(e)

    async def call_data_analysis_async(reasoning: str, task: str, config: RunnableConfig) -> str:
        """Async function for data analysis tool."""
        logger.info(f"Delegating to data analysis agent (async). Reasoning: {reasoning}, Task: {task}")
        import traceback

        sub_config = _build_sub_agent_config(config)
        try:
            response = await agent.ainvoke({"messages": task}, config=sub_config)
            return _extract_final_content(response)
        except GraphInterrupt as e:
            logger.info(f"Data analysis agent interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            logger.error(f"Run data analysis agent error:\n{repr(e)}")
            traceback.print_exc()
            return _format_data_analysis_error(e)

    description = """Delegate complex data analysis tasks to a specialized sub-agent.
Use this tool for:
1. Single metric trend forecasting
2. Single metric anomaly detection
3. Single metric anomaly drill-down (root cause analysis)
4. Multi-metric correlation
5. Business combination analysis

Provide a detailed, self-contained task description including metrics, dimensions, time ranges, and the
specific analysis goal. IMPORTANT: this sub-agent does NOT know the current time or the conversation
context, so resolve all relative time expressions (e.g. "last week", "yesterday") into concrete absolute
dates before delegating, and make the task unambiguous."""

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
