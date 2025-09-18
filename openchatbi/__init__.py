"""OpenChatBI core module initialization."""

import os

from langgraph.graph.state import CompiledStateGraph

from openchatbi.config_loader import ConfigLoader

# Global configuration instance
config = ConfigLoader()
# Skip config loading during documentation build
if not os.environ.get("SPHINX_BUILD"):
    config.load()
else:
    config.set({})


def get_default_graph():
    """
    Build the synchronous mode of the agent graph using default catalog in config.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    if os.environ.get("SPHINX_BUILD"):
        return None

    from langgraph.checkpoint.memory import MemorySaver
    from openchatbi.agent_graph import build_agent_graph_sync
    from openchatbi.tool.memory import get_sync_memory_store

    checkpointer = MemorySaver()
    return build_agent_graph_sync(
        config.get().catalog_store, checkpointer=checkpointer, memory_store=get_sync_memory_store()
    )
