"""Tests for memory tool functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_openai import ChatOpenAI

# Check if pysqlite3 is available, if not skip these tests
pysqlite3 = pytest.importorskip("pysqlite3", reason="pysqlite3 not available")

from openchatbi.tool.memory import (
    StructuredToolWithRequired,
    UserProfile,
    cleanup_async_memory_store,
    fix_schema_for_openai,
    get_async_memory_store,
    get_async_memory_tools,
    get_memory_manager,
    get_memory_tools,
    get_sync_memory_store,
    setup_async_memory_store,
)


class TestUserProfile:
    """Test UserProfile model functionality."""

    def test_user_profile_basic_initialization(self):
        """Test basic UserProfile model creation."""
        profile = UserProfile(name="John Doe", language="English", timezone="UTC", jargon="Technical")

        assert profile.name == "John Doe"
        assert profile.language == "English"
        assert profile.timezone == "UTC"
        assert profile.jargon == "Technical"

    def test_user_profile_optional_fields(self):
        """Test UserProfile with optional fields."""
        profile = UserProfile()

        assert profile.name is None
        assert profile.language is None
        assert profile.timezone is None
        assert profile.jargon is None

    def test_user_profile_partial_initialization(self):
        """Test UserProfile with partial field initialization."""
        profile = UserProfile(name="Jane Smith", language="Spanish")

        assert profile.name == "Jane Smith"
        assert profile.language == "Spanish"
        assert profile.timezone is None
        assert profile.jargon is None

    def test_user_profile_serialization(self):
        """Test UserProfile model serialization."""
        profile = UserProfile(name="Test User", timezone="EST")

        data = profile.model_dump()
        assert data["name"] == "Test User"
        assert data["timezone"] == "EST"
        assert data["language"] is None
        assert data["jargon"] is None


class TestMemoryStoreManagement:
    """Test memory store management functions."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self, tmp_path: Path):
        """Setup test environment with temporary database."""
        self.temp_db_path = tmp_path / "test_memory.db"
        # Clean up any global state
        import openchatbi.tool.memory as memory_module

        memory_module.sync_memory_store = None
        memory_module.async_memory_store = None
        memory_module.async_store_context_manager = None

    @patch("openchatbi.tool.memory.sqlite3.connect")
    @patch("openchatbi.tool.memory.config.get")
    def test_get_sync_memory_store(self, mock_config, mock_connect):
        """Test sync memory store creation."""
        mock_config.return_value.embedding_model = Mock()
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock SqliteStore
        with patch("openchatbi.tool.memory.SqliteStore") as mock_store_class:
            mock_store = Mock()
            mock_store_class.return_value = mock_store

            store = get_sync_memory_store()

            assert store == mock_store
            mock_store_class.assert_called_once()
            mock_store.setup.assert_called_once()

    @pytest.mark.asyncio
    @patch("openchatbi.tool.memory.AsyncSqliteStore.from_conn_string")
    @patch("openchatbi.tool.memory.config.get")
    async def test_get_async_memory_store(self, mock_config, mock_from_conn_string):
        """Test async memory store creation."""
        mock_config.return_value.embedding_model = Mock()

        # Mock the async context manager
        mock_context_manager = AsyncMock()
        mock_store = Mock()
        mock_context_manager.__aenter__.return_value = mock_store
        mock_from_conn_string.return_value = mock_context_manager

        store = await get_async_memory_store()

        assert store == mock_store
        mock_from_conn_string.assert_called_once()
        mock_context_manager.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    @patch("openchatbi.tool.memory.async_memory_store", new=Mock())
    @patch("openchatbi.tool.memory.async_store_context_manager")
    async def test_cleanup_async_memory_store(self, mock_context_manager):
        """Test async memory store cleanup."""
        mock_context_manager.__aexit__ = AsyncMock()

        await cleanup_async_memory_store()

        mock_context_manager.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    @patch("openchatbi.tool.memory.get_async_memory_store")
    async def test_setup_async_memory_store(self, mock_get_store):
        """Test async memory store setup."""
        mock_store = Mock()
        mock_get_store.return_value = mock_store

        result = await setup_async_memory_store()

        mock_get_store.assert_called_once()
        assert result is None


class TestMemoryTools:
    """Test memory tools creation and management."""

    @patch("openchatbi.tool.memory.create_manage_memory_tool")
    @patch("openchatbi.tool.memory.create_search_memory_tool")
    @patch("openchatbi.tool.memory.get_sync_memory_store")
    def test_get_memory_tools_sync_mode(self, mock_get_store, mock_search_tool, mock_manage_tool):
        """Test getting memory tools in sync mode."""
        mock_llm = FakeListChatModel(responses=["test"])
        mock_store = Mock()
        mock_get_store.return_value = mock_store

        mock_manage = Mock()
        mock_search = Mock()
        mock_manage_tool.return_value = mock_manage
        mock_search_tool.return_value = mock_search

        manage_tool, search_tool = get_memory_tools(mock_llm, sync_mode=True)

        assert manage_tool == mock_manage
        assert search_tool == mock_search
        mock_manage_tool.assert_called_once_with(namespace=("memories", "{user_id}"), store=mock_store)
        mock_search_tool.assert_called_once_with(namespace=("memories", "{user_id}"), store=mock_store)

    @patch("openchatbi.tool.memory.create_manage_memory_tool")
    @patch("openchatbi.tool.memory.create_search_memory_tool")
    def test_get_memory_tools_async_mode(self, mock_search_tool, mock_manage_tool):
        """Test getting memory tools in async mode."""
        mock_llm = FakeListChatModel(responses=["test"])

        mock_manage = Mock()
        mock_search = Mock()
        mock_manage_tool.return_value = mock_manage
        mock_search_tool.return_value = mock_search

        manage_tool, search_tool = get_memory_tools(mock_llm, sync_mode=False)

        assert manage_tool == mock_manage
        assert search_tool == mock_search
        mock_manage_tool.assert_called_once_with(namespace=("memories", "{user_id}"), store=None)
        mock_search_tool.assert_called_once_with(namespace=("memories", "{user_id}"), store=None)

    @patch("openchatbi.tool.memory.create_manage_memory_tool")
    @patch("openchatbi.tool.memory.create_search_memory_tool")
    def test_get_memory_tools_with_openai_llm(self, mock_search_tool, mock_manage_tool):
        """Test getting memory tools with OpenAI LLM (requires structured tool wrapper)."""
        mock_llm = Mock(spec=ChatOpenAI)

        mock_manage = Mock()
        mock_search = Mock()
        mock_manage_tool.return_value = mock_manage
        mock_search_tool.return_value = mock_search

        with patch("openchatbi.tool.memory.StructuredToolWithRequired") as mock_wrapper:
            mock_wrapped_manage = Mock()
            mock_wrapped_search = Mock()
            mock_wrapper.side_effect = [mock_wrapped_manage, mock_wrapped_search]

            manage_tool, search_tool = get_memory_tools(mock_llm)

            assert manage_tool == mock_wrapped_manage
            assert search_tool == mock_wrapped_search
            assert mock_wrapper.call_count == 2

    @pytest.mark.asyncio
    @patch("openchatbi.tool.memory.get_async_memory_store")
    @patch("openchatbi.tool.memory.get_memory_tools")
    async def test_get_async_memory_tools(self, mock_get_tools, mock_get_store):
        """Test getting async memory tools."""
        mock_llm = FakeListChatModel(responses=["test"])
        mock_store = Mock()
        mock_get_store.return_value = mock_store

        mock_manage = Mock()
        mock_search = Mock()
        mock_get_tools.return_value = (mock_manage, mock_search)

        manage_tool, search_tool = await get_async_memory_tools(mock_llm)

        assert manage_tool == mock_manage
        assert search_tool == mock_search
        mock_get_store.assert_called_once()
        mock_get_tools.assert_called_once_with(mock_llm, sync_mode=False, store=mock_store)


class TestMemoryManager:
    """Test memory manager functionality."""

    @patch("openchatbi.tool.memory.create_memory_store_manager")
    @patch("openchatbi.tool.memory.config.get")
    def test_get_memory_manager(self, mock_config, mock_create_manager):
        """Test memory manager creation."""
        mock_llm = Mock()
        mock_config.return_value.default_llm = mock_llm
        mock_manager = Mock()
        mock_create_manager.return_value = mock_manager

        manager = get_memory_manager()

        assert manager == mock_manager
        mock_create_manager.assert_called_once_with(
            mock_llm,
            schemas=[UserProfile],
            instructions="Extract user profile information",
            enable_inserts=False,
        )

    @patch("openchatbi.tool.memory.memory_manager", new=Mock())
    @patch("openchatbi.tool.memory.create_memory_store_manager")
    @patch("openchatbi.tool.memory.config.get")
    def test_get_memory_manager_singleton(self, mock_config, mock_create_manager):
        """Test memory manager singleton behavior."""
        # Reset the global variable for this test
        import openchatbi.tool.memory as memory_module

        existing_manager = Mock()
        memory_module.memory_manager = existing_manager

        manager = get_memory_manager()

        # Should return existing manager without creating new one
        assert manager == existing_manager
        mock_create_manager.assert_not_called()


class TestSchemaFixer:
    """Test schema fixing functionality for OpenAI compatibility."""

    def test_fix_schema_for_openai_basic(self):
        """Test basic schema fixing."""
        schema = {"properties": {"field1": {"type": "string"}, "field2": {"type": "number"}}}

        fix_schema_for_openai(schema)

        assert schema["required"] == ["field1", "field2"]

    def test_fix_schema_for_openai_nested_object(self):
        """Test schema fixing with nested objects."""
        schema = {
            "properties": {
                "nested": {"type": "object", "additionalProperties": True, "properties": {"inner": {"type": "string"}}}
            }
        }

        fix_schema_for_openai(schema)

        assert schema["required"] == ["nested"]
        assert schema["properties"]["nested"]["additionalProperties"] is False

    def test_fix_schema_for_openai_with_arrays(self):
        """Test schema fixing with array properties."""
        schema = {"properties": {"items": {"type": "array", "items": {"type": "object", "additionalProperties": True}}}}

        fix_schema_for_openai(schema)

        assert schema["required"] == ["items"]
        assert schema["properties"]["items"]["items"]["additionalProperties"] is False


class TestStructuredToolWithRequired:
    """Test StructuredToolWithRequired wrapper functionality."""

    def test_structured_tool_with_required_initialization(self):
        """Test StructuredToolWithRequired initialization."""
        mock_original_tool = Mock()
        mock_original_tool.name = "test_tool"
        mock_original_tool.description = "Test description"
        mock_original_tool.args_schema = Mock()
        mock_original_tool.func = Mock()
        mock_original_tool.coroutine = None

        with patch("openchatbi.tool.memory.StructuredTool.__init__", return_value=None) as mock_init:
            wrapper = StructuredToolWithRequired(mock_original_tool)

            # Verify the __init__ was called with correct parameters
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            assert call_args.kwargs["name"] == "test_tool"
            assert call_args.kwargs["description"] == "Test description"

    def test_tool_call_schema_property(self):
        """Test tool_call_schema cached property."""
        mock_original_tool = Mock()
        mock_original_tool.name = "test_tool"
        mock_original_tool.description = "Test description"
        mock_original_tool.args_schema = Mock()
        mock_original_tool.func = Mock()
        mock_original_tool.coroutine = None

        with patch("openchatbi.tool.memory.StructuredTool.__init__", return_value=None):
            wrapper = StructuredToolWithRequired(mock_original_tool)

            # Mock the parent's tool_call_schema
            mock_tcs = Mock()
            mock_tcs.model_config = {}

            with patch("openchatbi.tool.memory.StructuredTool.tool_call_schema", new_callable=lambda: mock_tcs):
                result = wrapper.tool_call_schema

                assert result == mock_tcs
                assert "json_schema_extra" in mock_tcs.model_config
