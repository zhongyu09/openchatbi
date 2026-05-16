"""Persistence compatibility checks for LangGraph v1.1 migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from openchatbi.tool.memory import get_memory_tools, get_sync_memory_store


def test_sqlite_store_reads_pre_existing_memory_data(tmp_path: Path) -> None:
    """A SQLite memory database created before reopening remains readable."""
    db_path = tmp_path / "memory.db"
    namespace = ("memories", "user-1")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.isolation_level = None
    store = SqliteStore(conn)
    store.setup()
    store.put(namespace, "profile", {"text": "prefers bar charts"}, index=False)
    conn.close()

    reopened_conn = sqlite3.connect(db_path, check_same_thread=False)
    reopened_conn.isolation_level = None
    reopened_store = SqliteStore(reopened_conn)
    reopened_store.setup()
    items = reopened_store.search(namespace, limit=10)
    reopened_conn.close()

    assert [item.key for item in items] == ["profile"]
    assert items[0].value == {"text": "prefers bar charts"}


def test_sqlite_checkpointer_can_reopen_existing_database(tmp_path: Path) -> None:
    """Checkpoint SQLite files should be reusable across context manager lifecycles."""
    db_path = tmp_path / "checkpoints.db"

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        saver.setup()

    with SqliteSaver.from_conn_string(str(db_path)) as reopened:
        reopened.setup()
        assert list(reopened.list({"configurable": {"thread_id": "missing-thread"}})) == []


def test_memory_store_disabled_when_embedding_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No embedding model remains a supported memory-disabled configuration."""
    import openchatbi.tool.memory as memory_module

    memory_module.sync_memory_store = None
    mock_config = Mock()
    mock_config.embedding_model = None
    monkeypatch.setattr(memory_module.config, "get", lambda: mock_config)

    assert get_sync_memory_store() is None
    assert get_memory_tools(Mock(), sync_mode=True) is None
