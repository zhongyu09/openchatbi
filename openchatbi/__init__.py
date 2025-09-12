"""OpenChatBI core module initialization."""

from langgraph.graph.state import CompiledStateGraph

from openchatbi.config_loader import ConfigLoader

# Global configuration instance
config = ConfigLoader()
config.load()


def get_default_graph() -> CompiledStateGraph:
    """
    Build the synchronous mode of the agent graph using default catalog in config.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from openchatbi.agent_graph import build_agent_graph_sync
    from openchatbi.tool.memory import get_sync_memory_store

    checkpointer = MemorySaver()
    return build_agent_graph_sync(
        config.get().catalog_store, checkpointer=checkpointer, memory_store=get_sync_memory_store()
    )
