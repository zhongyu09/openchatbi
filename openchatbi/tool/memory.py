import functools
import sys
from typing import Any

import pysqlite3 as sqlite3

# make sure langgraph sqlite connector uses pysqlite3
sys.modules["sqlite3"] = sqlite3

from langchain.tools import StructuredTool
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

try:
    from pydantic import BaseModel, ConfigDict
except ImportError:
    ConfigDict = None

# Use AsyncSqliteStore for async operations
async_memory_store = None
async_store_context_manager = None
sync_memory_store = None
memory_manager = None


# Define profile structure
class UserProfile(BaseModel):
    """Represents the full representation of a user."""

    name: str | None = None
    language: str | None = None
    timezone: str | None = None
    jargon: str | None = None


def get_sync_memory_store() -> SqliteStore:
    global sync_memory_store
    if sync_memory_store is None:
        # For backwards compatibility and sync operations
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        conn.isolation_level = None
        sync_memory_store = SqliteStore(
            conn,
            index={
                "dims": 1536,
                "embed": config.get().embedding_model,
                "fields": ["text"],  # specify which fields to embed
            },
        )
        try:
            sync_memory_store.setup()
        except Exception:
            pass
    return sync_memory_store


async def get_async_memory_store() -> AsyncSqliteStore:
    """Get or create the async memory store."""
    global async_memory_store, async_store_context_manager
    if async_memory_store is None:
        # AsyncSqliteStore.from_conn_string returns an async context manager
        async_store_context_manager = AsyncSqliteStore.from_conn_string(
            "memory.db",
            index={
                "dims": 1536,
                "embed": config.get().embedding_model,
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


class StructuredToolWithRequired(StructuredTool):
    def __init__(self, orig_tool: StructuredTool):
        name = getattr(orig_tool, "name", None)
        super().__init__(
            name=name,
            description=orig_tool.description,
            args_schema=orig_tool.args_schema,
            func=orig_tool.func,
            coroutine=orig_tool.coroutine,
        )

    @functools.cached_property
    def tool_call_schema(self) -> "ArgsSchema":
        tcs = super().tool_call_schema
        try:
            if tcs.model_config:
                tcs.model_config["json_schema_extra"] = fix_schema_for_openai
            elif ConfigDict is not None:
                tcs.model_config = ConfigDict(json_schema_extra=fix_schema_for_openai)
        except Exception:
            pass
        return tcs


def get_memory_tools(
    llm: BaseChatModel, sync_mode: bool = False, store: Any | None = None
) -> tuple[StructuredTool, StructuredTool]:
    # Get the appropriate store based on mode
    if not store:
        if sync_mode:
            store = get_sync_memory_store()
        else:
            # For async mode, pass None to let langmem handle store internally
            store = None

    # create langmem manage memory tool with {user_id} template
    manage_memory_tool = create_manage_memory_tool(namespace=("memories", "{user_id}"), store=store)
    search_memory_tool = create_search_memory_tool(namespace=("memories", "{user_id}"), store=store)

    if isinstance(llm, BaseChatOpenAI):
        manage_memory_tool = StructuredToolWithRequired(manage_memory_tool)
        search_memory_tool = StructuredToolWithRequired(search_memory_tool)
    return manage_memory_tool, search_memory_tool


async def get_async_memory_tools(llm: BaseChatModel) -> tuple[StructuredTool, StructuredTool]:
    """Get memory tools configured with async store."""
    async_store = await get_async_memory_store()
    return get_memory_tools(llm, sync_mode=False, store=async_store)
