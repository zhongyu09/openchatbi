"""Common AsyncGraphManager for UIs."""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from openchatbi import config
from openchatbi.agent_graph import build_agent_graph_async
from openchatbi.llm.llm import get_default_llm
from openchatbi.tool.memory import cleanup_async_memory_store, get_async_memory_tools, setup_async_memory_store
from openchatbi.utils import log


class AsyncGraphManager:
    """Manages the async graph and checkpointer lifecycle"""

    def __init__(self):
        self.checkpointer = None
        self.graph = None
        self._context_manager = None
        self._initialized = False

    async def initialize(self):
        """Initialize the graph and checkpointer"""
        if self._initialized:
            return

        try:
            # Setup async memory store
            await setup_async_memory_store()

            # Initialize checkpointer
            self._context_manager = AsyncSqliteSaver.from_conn_string("checkpoints.db")
            self.checkpointer = await self._context_manager.__aenter__()

            # Get async memory tools
            from openchatbi.tool.memory import get_async_memory_store

            async_store = await get_async_memory_store()
            async_memory_tools = await get_async_memory_tools(get_default_llm())

            # Build the graph
            self.graph = await build_agent_graph_async(
                config.get().catalog_store,
                checkpointer=self.checkpointer,
                memory_store=async_store,
                memory_tools=async_memory_tools,
            )

            self._initialized = True
            log("Graph initialized successfully")

        except Exception as e:
            log(f"Failed to initialize graph: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.checkpointer is not None and self._context_manager is not None:
            try:
                await self._context_manager.__aexit__(None, None, None)
                await cleanup_async_memory_store()
                log("Graph cleaned up successfully")
            except Exception as e:
                log(f"Error during cleanup: {e}")
            finally:
                self.checkpointer = None
                self.graph = None
                self._context_manager = None
                self._initialized = False
