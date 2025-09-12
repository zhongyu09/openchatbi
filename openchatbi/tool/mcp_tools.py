"""MCP (Model Context Protocol) tools integration for OpenChatBI.

This module provides integration with MCP servers using langchain-mcp-adapters,
allowing the agent to use external tools through the Model Context Protocol.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from openchatbi.constants import MCP_TOOL_DEFAULT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def make_tool_sync_compatible(tool: StructuredTool, timeout: int) -> StructuredTool:
    """Make an async-only StructuredTool compatible with sync invocation.

    This wraps the async coroutine with a sync function that runs it in an event loop.

    Args:
        tool: The StructuredTool to make sync-compatible
        timeout: Timeout in seconds for tool execution

    Returns:
        StructuredTool with sync compatibility
    """
    if tool.func is not None:
        # Tool already has sync support
        return tool

    if tool.coroutine is None:
        # Tool has no async function either, can't help
        return tool

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper for async tool function."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, can't use run_until_complete
                # Create a new thread with its own event loop
                with ThreadPoolExecutor(max_workers=1) as executor:

                    def run_in_new_loop() -> Any:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(tool.coroutine(*args, **kwargs))  # type: ignore
                        finally:
                            new_loop.close()

                    future = executor.submit(run_in_new_loop)
                    return future.result(timeout=timeout)
            else:
                # No running loop, we can use run_until_complete
                return loop.run_until_complete(tool.coroutine(*args, **kwargs))  # type: ignore
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(tool.coroutine(*args, **kwargs))  # type: ignore
            finally:
                loop.close()

    # Create a new StructuredTool with both sync and async functions
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=sync_wrapper,
        coroutine=tool.coroutine,
    )


class MCPServerConfig(BaseModel):
    """Configuration for MCP server connection."""

    name: str = Field(description="Name of the MCP server")
    transport: str = Field(default="stdio", description="Transport type: stdio, sse, or streamable_http")

    # For stdio transport
    command: list[str] = Field(default_factory=list, description="Command to start the MCP server")
    args: list[str] = Field(default_factory=list, description="Arguments for the MCP server")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")

    # For HTTP transports (sse, streamable_http)
    url: str = Field(default="", description="URL for HTTP-based transports")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")

    # Common settings
    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")
    timeout: int = Field(default=MCP_TOOL_DEFAULT_TIMEOUT_SECONDS, description="Connection timeout in seconds")


async def create_mcp_tools_async(server_configs: list[dict[str, Any]]) -> list[StructuredTool]:
    """Create MCP tools asynchronously from server configurations.

    This function processes MCP server configurations, establishes connections to enabled
    servers, retrieves available tools, and makes them sync-compatible with proper
    timeout configuration.

    Args:
        server_configs: List of MCP server configuration dictionaries containing
                       server connection details, transport settings, and timeouts

    Returns:
        List of LangChain StructuredTool instances with mcp_ prefixes and sync compatibility
    """
    if not server_configs:
        return []

    # Filter enabled servers and convert to MCPServerConfig
    enabled_servers = {}
    max_timeout = MCP_TOOL_DEFAULT_TIMEOUT_SECONDS  # Default from constants

    for config_dict in server_configs:
        try:
            config = MCPServerConfig(**config_dict)
            if not config.enabled:
                continue

            server_name = config.name

            # Track the maximum timeout across all servers
            max_timeout = max(max_timeout, config.timeout)

            # Build server configuration for MultiServerMCPClient
            if config.transport == "stdio":
                if not config.command:
                    logger.warning(f"MCP server {server_name}: command required for stdio transport")
                    continue

                enabled_servers[server_name] = {
                    "transport": "stdio",
                    "command": config.command[0] if config.command else "",
                    "args": config.command[1:] + config.args if len(config.command) > 1 else config.args,
                    "env": config.env,
                }
            elif config.transport in ["sse", "streamable_http"]:
                if not config.url:
                    logger.warning(f"MCP server {server_name}: url required for {config.transport} transport")
                    continue

                server_config: dict[str, Any] = {
                    "transport": config.transport,
                    "url": config.url,
                }
                if config.headers:
                    server_config["headers"] = config.headers
                enabled_servers[server_name] = server_config
            else:
                logger.warning(f"MCP server {server_name}: unsupported transport {config.transport}")
                continue

        except Exception as e:
            logger.error(f"Invalid MCP server configuration: {e}")
            continue

    if not enabled_servers:
        logger.info("No enabled MCP servers found")
        return []

    try:
        # Create MultiServerMCPClient and get tools with timeout
        client = MultiServerMCPClient(enabled_servers)
        tools = await asyncio.wait_for(client.get_tools(), timeout=max_timeout)

        logger.info(f"Successfully loaded {len(tools)} MCP tools from {len(enabled_servers)} servers")

        # Add server prefix to tool names and make sync-compatible
        prefixed_tools = []
        for tool in tools:
            # Get server name from tool metadata or guess from tool name
            original_name = tool.name
            if not original_name.startswith("mcp_"):
                tool.name = f"mcp_{original_name}"

            # Make tool sync-compatible with configured timeout
            sync_compatible_tool = make_tool_sync_compatible(tool, timeout=max_timeout)
            prefixed_tools.append(sync_compatible_tool)

        return prefixed_tools

    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        return []


def create_mcp_tools_sync(server_configs: list[dict[str, Any]]) -> list[StructuredTool]:
    """Create MCP tools from server configurations synchronously.

    This function initializes MCP tools in a separate thread with its own event loop
    to avoid conflicts with existing async contexts.

    Args:
        server_configs: List of MCP server configuration dictionaries

    Returns:
        List of LangChain StructuredTool instances with sync compatibility
    """

    if not server_configs:
        return []

    # For sync mode, run async initialization in a thread
    def sync_initialize() -> list[StructuredTool]:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(create_mcp_tools_async(server_configs))
        except Exception as e:
            logger.error(f"Failed to create MCP tools in sync mode: {e}")
            return []
        finally:
            loop.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(sync_initialize)
            return future.result(timeout=MCP_TOOL_DEFAULT_TIMEOUT_SECONDS)
    except Exception as e:
        logger.error(f"MCP tools sync initialization failed: {e}")
        return []


# Global variable to store async-initialized tools
_async_mcp_tools = None


async def get_mcp_tools_async(server_configs: list[dict[str, Any]]) -> list[StructuredTool]:
    """Get MCP tools asynchronously, using cached version if available.

    Args:
        server_configs: List of MCP server configuration dictionaries

    Returns:
        List of cached or newly created LangChain StructuredTool instances
    """
    global _async_mcp_tools

    if _async_mcp_tools is None:
        _async_mcp_tools = await create_mcp_tools_async(server_configs)

    return _async_mcp_tools


def reset_mcp_tools_cache() -> None:
    """Reset the async MCP tools cache."""
    global _async_mcp_tools
    _async_mcp_tools = None
