"""Tests for langmem decay reranking + importance/last_used/use_count stamping."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import FakeListChatModel

pytest.importorskip("pysqlite3", reason="pysqlite3 not available")

from openchatbi.tool.memory import (  # noqa: E402
    _rerank_search_results,
    _stamp_memory_value,
    get_memory_tools,
)


def _item(text, days_ago, importance=1.0, use_count=0, score=0.5):
    it = Mock()
    it.value = {
        "text": text,
        "importance": importance,
        "use_count": use_count,
        "last_used": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
    }
    it.score = score
    return it


def test_rerank_orders_by_composite_score():
    fresh = _item("fresh", days_ago=1, score=0.5)
    stale = _item("stale", days_ago=200, score=0.5)
    out = _rerank_search_results([stale, fresh])
    assert out[0].value["text"] == "fresh"  # recency-decayed composite floats fresh up


def test_rerank_tolerates_plain_dicts():
    a = {"value": {"text": "a", "last_used": datetime.now(timezone.utc).isoformat()}, "score": 0.9}
    b = {"value": {"text": "b", "last_used": (datetime.now(timezone.utc) - timedelta(days=300)).isoformat()}, "score": 0.9}
    out = _rerank_search_results([b, a])
    assert out[0]["value"]["text"] == "a"


def test_stamp_memory_value_adds_provenance_fields():
    stamped = _stamp_memory_value({"text": "remember X"})
    assert stamped["importance"] == 1.0
    assert stamped["use_count"] == 0
    assert "last_used" in stamped


@patch("openchatbi.tool.memory.get_memory_config")
@patch("openchatbi.tool.memory.create_manage_memory_tool")
@patch("openchatbi.tool.memory.create_search_memory_tool")
@patch("openchatbi.tool.memory.get_sync_memory_store")
def test_rerank_disabled_by_default_returns_raw_tools(
    mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg
):
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
def test_rerank_enabled_wraps_search_tool(
    mock_get_store, mock_search_tool, mock_manage_tool, mock_get_cfg
):
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
