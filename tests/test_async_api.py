"""Regression tests for the sample async API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
async def test_get_user_memories_uses_async_store_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async endpoint must not call the store's event-loop-blocking sync API."""
    from openchatbi.tool import memory as memory_module
    from sample_api.async_api import get_user_memories

    item = SimpleNamespace(
        key="profile",
        value={"text": "prefers bar charts"},
        created_at="2026-08-15T00:00:00Z",
        updated_at="2026-08-15T00:00:00Z",
    )
    memory_store = Mock()
    memory_store.asearch = AsyncMock(return_value=[item])
    memory_store.search.side_effect = AssertionError("synchronous search must not be called")

    async def get_store():
        return memory_store

    monkeypatch.setattr(memory_module, "get_async_memory_store", get_store)

    response = await get_user_memories("user-1")

    memory_store.asearch.assert_awaited_once_with(("memories", "user-1"))
    memory_store.search.assert_not_called()
    assert response == {
        "user_id": "user-1",
        "total_memories": 1,
        "memories": [
            {
                "key": "profile",
                "content": {"text": "prefers bar charts"},
                "namespace": "('memories', 'user-1')",
                "created_at": "2026-08-15T00:00:00Z",
                "updated_at": "2026-08-15T00:00:00Z",
            }
        ],
    }
