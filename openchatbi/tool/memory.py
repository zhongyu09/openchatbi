import functools
import logging
import sys
from typing import Any

# langgraph's sqlite modules must see the selected sqlite module before import.
# ruff: noqa: E402, I001
try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover
    import sqlite3

# Make sure langgraph sqlite connector uses the same sqlite module.
sys.modules["sqlite3"] = sqlite3

from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langmem import (
    create_manage_memory_tool,
    create_memory_store_manager,
    create_search_memory_tool,
)

from openchatbi import config
from openchatbi.memory_config import get_memory_config
from openchatbi.memory_scoring import composite_score

try:
    from pydantic import BaseModel, ConfigDict
except ImportError:
    ConfigDict = None  # type: ignore[assignment,misc]

# Use AsyncSqliteStore for async operations
async_memory_store = None
async_store_context_manager = None
sync_memory_store = None
memory_manager = None
logger = logging.getLogger(__name__)


# Define profile structure
class UserProfile(BaseModel):
    """Represents the full representation of a user."""

    name: str | None = None
    language: str | None = None
    timezone: str | None = None
    jargon: str | None = None


def get_sync_memory_store() -> SqliteStore | None:
    global sync_memory_store
    embedding_model = config.get().embedding_model
    if not embedding_model:
        return None
    if sync_memory_store is None:
        # For backwards compatibility and sync operations
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        conn.isolation_level = None
        sync_memory_store = SqliteStore(
            conn,
            index={
                "dims": 1536,
                "embed": embedding_model,  # type: ignore[typeddict-item]
                "fields": ["text"],  # specify which fields to embed
            },
        )
        try:
            sync_memory_store.setup()
        except Exception as e:
            logger.warning("Memory store setup failed; sync memory tools may be unavailable: %s", e)
    return sync_memory_store


async def get_async_memory_store() -> AsyncSqliteStore | None:
    """Get or create the async memory store."""
    global async_memory_store, async_store_context_manager
    embedding_model = config.get().embedding_model
    if not embedding_model:
        return None
    if async_memory_store is None:
        # AsyncSqliteStore.from_conn_string returns an async context manager
        async_store_context_manager = AsyncSqliteStore.from_conn_string(
            "memory.db",
            index={
                "dims": 1536,
                "embed": embedding_model,  # type: ignore[typeddict-item]
                "fields": ["text"],  # specify which fields to embed
            },
        )
        async_memory_store = await async_store_context_manager.__aenter__()
    return async_memory_store


async def cleanup_async_memory_store() -> None:
    """Cleanup async memory store resources."""
    global async_memory_store, async_store_context_manager
    if async_memory_store is not None and async_store_context_manager is not None:
        try:
            await async_store_context_manager.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error cleaning up async memory store: {e}")
        finally:
            async_memory_store = None
            async_store_context_manager = None


async def setup_async_memory_store() -> Any:
    """Setup async memory store for langmem."""
    await get_async_memory_store()


def fix_schema_for_openai(schema: dict) -> None:
    props = schema.get("properties", {})
    schema["required"] = list(props.keys())

    # Since Pydantic 2.11, it will always add `additionalProperties: True` for arbitrary dictionary schemas
    # If it is already set to True, we need override it to False
    # Can remove this fix when the patch release: https://github.com/langchain-ai/langchain/pull/32879
    def fix(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "additionalProperties" in obj and obj["additionalProperties"]:
                obj["additionalProperties"] = False
            for v in obj.values():
                fix(v)
        elif isinstance(obj, list):
            for item in obj:
                fix(item)

    fix(schema)


def get_memory_manager() -> Any:
    global memory_manager
    if memory_manager is None:
        memory_manager = create_memory_store_manager(
            config.get().default_llm,
            schemas=[UserProfile],
            instructions="Extract user profile information",
            enable_inserts=False,
        )
    return memory_manager


def _item_value(item: Any) -> dict:
    """Extract the value dict from a langgraph Item or a plain dict result."""
    if isinstance(item, dict):
        return item.get("value", {}) or {}
    return getattr(item, "value", {}) or {}


def _item_base_score(item: Any) -> float:
    """Extract the retrieval similarity/score from an Item or dict, defaulting to 1.0."""
    if isinstance(item, dict):
        return float(item.get("score", 1.0) or 1.0)
    return float(getattr(item, "score", 1.0) or 1.0)


def _rerank_search_results(items: list) -> list:
    """Re-rank langmem search results by composite_score(similarity, importance, decay, use_count)."""
    cfg = get_memory_config()

    def _key(item: Any) -> float:
        value = _item_value(item)
        return composite_score(
            _item_base_score(item),
            float(value.get("importance", 1.0) or 1.0),
            value.get("last_used", ""),
            int(value.get("use_count", 0) or 0),
            cfg,
        )

    return sorted(items, key=_key, reverse=True)


class StructuredToolWithRequired(StructuredTool):
    def __init__(self, orig_tool: StructuredTool):
        name = getattr(orig_tool, "name", None) or ""
        super().__init__(
            name=name,
            description=orig_tool.description,
            args_schema=orig_tool.args_schema,
            func=orig_tool.func,
            coroutine=orig_tool.coroutine,
        )

    @functools.cached_property
    def tool_call_schema(self) -> Any:
        tcs = super().tool_call_schema
        try:
            if not isinstance(tcs, dict) and tcs.model_config:
                tcs.model_config["json_schema_extra"] = fix_schema_for_openai
            elif not isinstance(tcs, dict) and ConfigDict is not None:
                tcs.model_config = ConfigDict(json_schema_extra=fix_schema_for_openai)
        except Exception as e:
            logger.warning("Unable to attach OpenAI schema compatibility hook: %s", e)
        return tcs


def get_memory_tools(llm: BaseChatModel, sync_mode: bool = False, store: Any | None = None) -> list[BaseTool] | None:
    # Get the appropriate store based on mode
    if not store:
        if sync_mode:
            store = get_sync_memory_store()
        else:
            store = None
    if not store:
        return None

    # create langmem manage memory tool with {user_id} template
    manage_memory_tool = create_manage_memory_tool(namespace=("memories", "{user_id}"), store=store)
    search_memory_tool = create_search_memory_tool(namespace=("memories", "{user_id}"), store=store)

    if isinstance(llm, BaseChatOpenAI):
        manage_memory_tool = StructuredToolWithRequired(manage_memory_tool)
        search_memory_tool = StructuredToolWithRequired(search_memory_tool)

    mem_cfg = get_memory_config()
    if not getattr(mem_cfg, "enable_memory_decay_rerank", False):
        return [manage_memory_tool, search_memory_tool]

    _raw_search = search_memory_tool
    _raw_manage = manage_memory_tool

    @tool("search_memory", description=getattr(_raw_search, "description", "Search long-term memory."))
    def reranked_search_memory(query: str) -> Any:
        """Search long-term memory, re-ranked by importance/recency decay."""
        results = _raw_search.invoke({"query": query})
        if isinstance(results, list):
            return _rerank_search_results(results)
        return results

    @tool("manage_memory", description=getattr(_raw_manage, "description", "Create or update long-term memory."))
    def stamped_manage_memory(content: str) -> Any:
        """Create/update long-term memory, rerank-enabled wrapper (content passed through unchanged)."""
        return _raw_manage.invoke({"content": content})

    return [stamped_manage_memory, reranked_search_memory]


async def get_async_memory_tools(llm: BaseChatModel) -> list[BaseTool]:
    """Get memory tools configured with async store."""
    async_store = await get_async_memory_store()
    return get_memory_tools(llm, sync_mode=False, store=async_store) or []
