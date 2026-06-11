"""Tests for langmem decay reranking."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from langchain_core.language_models import FakeListChatModel

from openchatbi.tool.memory import (
    _rerank_search_results,
    get_memory_tools,
)


def _item(text, days_ago, importance=1.0, use_count=0, score=0.5):
    it = Mock()
    it.value = {
        "text": text,
        "importance": importance,
        "use_count": use_count,
        "last_used": (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(),
    }
    it.score = score
    return it


def test_rerank_orders_by_composite_score():
    fresh = _item("fresh", days_ago=1, score=0.5)
    stale = _item("stale", days_ago=200, score=0.5)
    out = _rerank_search_results([stale, fresh])
    assert out[0].value["text"] == "fresh"  # recency-decayed composite floats fresh up


def test_rerank_tolerates_plain_dicts():
    a = {"value": {"text": "a", "last_used": datetime.now(UTC).isoformat()}, "score": 0.9}
    b = {
        "value": {"text": "b", "last_used": (datetime.now(UTC) - timedelta(days=300)).isoformat()},
        "score": 0.9,
    }
    out = _rerank_search_results([b, a])
    assert out[0]["value"]["text"] == "a"


@patch("openchatbi.tool.memory.get_memory_config")
@patch("openchatbi.tool.memory.create_manage_memory_tool")
@patch("openchatbi.tool.memory.create_search_memory_tool")
@patch("openchatbi.tool.memory.get_sync_memory_store")
def test_rerank_disabled_by_default_returns_raw_tools(mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg):
    cfg = Mock()
    cfg.enable_memory_decay_rerank = False
    mock_get_cfg.return_value = cfg
    mock_get_store.return_value = Mock()
    raw_search = Mock()
    raw_manage = Mock()
    mock_search_tool.return_value = raw_search
    mock_manage_tool.return_value = raw_manage

    tools = get_memory_tools(FakeListChatModel(responses=["x"]), sync_mode=True)
    # default OFF: the original langmem tool objects are returned unwrapped
    assert tools[0] is raw_manage
    assert tools[1] is raw_search


@patch("openchatbi.tool.memory.get_memory_config")
@patch("openchatbi.tool.memory.create_manage_memory_tool")
@patch("openchatbi.tool.memory.create_search_memory_tool")
@patch("openchatbi.tool.memory.get_sync_memory_store")
def test_rerank_enabled_wraps_search_tool(mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg):
    cfg = Mock()
    cfg.enable_memory_decay_rerank = True
    mock_get_cfg.return_value = cfg
    mock_get_store.return_value = Mock()
    mock_search_tool.return_value = Mock()
    mock_manage_tool.return_value = Mock()

    tools = get_memory_tools(FakeListChatModel(responses=["x"]), sync_mode=True)
    # wrapped tools are new StructuredTool instances, not the raw mocks
    assert tools[0] is not mock_manage_tool.return_value
    assert tools[1] is not mock_search_tool.return_value


@patch("openchatbi.tool.memory.get_memory_config")
@patch("openchatbi.tool.memory.create_manage_memory_tool")
@patch("openchatbi.tool.memory.create_search_memory_tool")
@patch("openchatbi.tool.memory.get_sync_memory_store")
def test_manage_wrapper_passes_content_string_unchanged(
    mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg
):
    """manage wrapper must invoke raw tool with content=<original string>, not a dict."""
    cfg = Mock()
    cfg.enable_memory_decay_rerank = True
    mock_get_cfg.return_value = cfg
    mock_get_store.return_value = Mock()
    mock_search_tool.return_value = Mock()
    raw_manage = Mock()
    mock_manage_tool.return_value = raw_manage

    tools = get_memory_tools(FakeListChatModel(responses=["x"]), sync_mode=True)
    manage_wrapper = tools[0]

    manage_wrapper.invoke({"content": "remember this fact"})

    raw_manage.invoke.assert_called_once()
    call_kwargs = raw_manage.invoke.call_args[0][0]
    assert (
        call_kwargs["content"] == "remember this fact"
    ), f"Expected content to be the original string, got: {call_kwargs['content']!r}"
    assert isinstance(call_kwargs["content"], str), f"content must be a string, not {type(call_kwargs['content'])}"
