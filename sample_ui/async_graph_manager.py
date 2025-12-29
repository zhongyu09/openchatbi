"""Common AsyncGraphManager for UIs."""

from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from openchatbi import config
from openchatbi.agent_graph import build_agent_graph_async
from openchatbi.tool.memory import cleanup_async_memory_store, get_async_memory_store, setup_async_memory_store
from openchatbi.utils import log


class AsyncGraphManager:
    """Manages the async graph and checkpointer lifecycle"""

    def __init__(self):
        self.checkpointer = None
        self.graph = None  # Default graph (backwards compatible)
        self.graphs: dict[str, Any] = {}
        self._context_manager = None
        self._memory_store = None
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

            # Cache store for graph builds
            self._memory_store = await get_async_memory_store()

            self._initialized = True

            # Build default graph for backwards compatibility
            self.graph = await self.get_graph()

            log("Graph initialized successfully")

        except Exception as e:
            self._initialized = False
            log(f"Failed to initialize graph: {e}")
            raise

    async def get_graph(self, llm_provider: str | None = None):
        """Get or build a graph for the requested LLM provider."""
        if not self._initialized:
            await self.initialize()

        key = llm_provider or "__default__"
        if key in self.graphs:
            return self.graphs[key]

        graph = await build_agent_graph_async(
            config.get().catalog_store,
            checkpointer=self.checkpointer,
            memory_store=self._memory_store,
            memory_tools=None,  # Let graph builder create provider-appropriate tools
            llm_provider=llm_provider,
        )
        self.graphs[key] = graph
        return graph

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
                self.graphs = {}
                self._context_manager = None
                self._memory_store = None
                self._initialized = False
