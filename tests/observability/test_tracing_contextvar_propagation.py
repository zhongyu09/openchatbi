"""ContextVars set at turn start must survive the sync ToolNode / to_thread
boundary that the text2sql tool crosses (get_sql_tools does not thread config
to sql_graph.invoke at agent_graph.py:158/175)."""

import asyncio
import contextvars

from openchatbi.observability.context import get_run_context, set_run_context


def test_contextvar_survives_to_thread() -> None:
    set_run_context("alice", "req-7")

    def inner() -> tuple[str | None, str | None]:
        # Simulates execute_sql_node reading attribution inside the subgraph.
        return get_run_context()

    async def main() -> tuple[str | None, str | None]:
        # asyncio.to_thread copies the current context → ids must propagate.
        return await asyncio.to_thread(inner)

    assert asyncio.run(main()) == ("alice", "req-7")


def test_contextvar_survives_copy_context() -> None:
    # LangGraph's sync ToolNode runs nodes via contextvars.copy_context().run.
    set_run_context("bob", "req-9")
    ctx = contextvars.copy_context()
    assert ctx.run(get_run_context) == ("bob", "req-9")
