"""Tests for observability run-context contextvars."""

import asyncio

from openchatbi.observability.context import (
    current_request_id,
    current_user_id,
    get_run_context,
    set_run_context,
)


def test_defaults_are_none() -> None:
    assert current_user_id.get() is None
    assert current_request_id.get() is None
    assert get_run_context() == (None, None)


def test_set_run_context_roundtrips() -> None:
    set_run_context("alice", "req-123")
    assert get_run_context() == ("alice", "req-123")
    assert current_user_id.get() == "alice"
    assert current_request_id.get() == "req-123"


def test_set_run_context_isolated_per_task() -> None:
    async def worker(uid: str) -> tuple[str | None, str | None]:
        set_run_context(uid, f"req-{uid}")
        await asyncio.sleep(0)
        return get_run_context()

    async def main() -> list[tuple[str | None, str | None]]:
        return await asyncio.gather(worker("u1"), worker("u2"))

    results = asyncio.run(main())
    assert ("u1", "req-u1") in results
    assert ("u2", "req-u2") in results
