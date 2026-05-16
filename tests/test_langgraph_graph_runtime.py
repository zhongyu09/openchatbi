"""Regression tests for OpenChatBI graph runtime on LangGraph v1.1."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from openchatbi.agent_graph import build_agent_graph_async, build_agent_graph_sync, get_sql_tools


class DummyLLM:
    """Minimal LLM double used only for graph construction."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self


class FakeSQLGraph:
    """Compiled graph double that preserves dict-like invoke outputs."""

    def invoke(self, payload: dict) -> dict:
        assert payload == {"messages": "show revenue"}
        return {"sql": "SELECT 1", "data": "value\n1", "visualization_dsl": {"chart_type": "bar"}}

    async def ainvoke(self, payload: dict) -> dict:
        assert payload == {"messages": "show revenue"}
        return {"sql": "SELECT 1", "data": "value\n1", "visualization_dsl": {"chart_type": "bar"}}


def _patch_agent_graph_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import openchatbi.agent_graph as agent_graph

    monkeypatch.setattr(agent_graph, "get_llm", lambda provider=None: DummyLLM())
    monkeypatch.setattr(agent_graph, "build_sql_graph", lambda *args, **kwargs: FakeSQLGraph())
    monkeypatch.setattr(agent_graph, "get_memory_tools", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_graph, "create_mcp_tools_sync", lambda servers: [])
    monkeypatch.setattr(agent_graph, "get_mcp_tools_async", lambda servers: _async_empty_tools())
    monkeypatch.setattr(agent_graph, "check_forecast_service_health", lambda: False)


async def _async_empty_tools() -> list:
    return []


def test_sync_agent_graph_builds_with_langgraph_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_agent_graph_dependencies(monkeypatch)

    graph = build_agent_graph_sync(Mock(), checkpointer=None, memory_store=None, enable_context_management=False)

    assert graph.name == "agent_graph"
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_default_graph_entrypoint_uses_sync_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    import openchatbi
    import openchatbi.agent_graph as agent_graph
    import openchatbi.tool.memory as memory_module

    expected_graph = Mock(name="compiled_graph")
    mock_config = Mock()
    mock_config.catalog_store = Mock(name="catalog_store")

    monkeypatch.setattr(openchatbi.config, "get", lambda: mock_config)
    monkeypatch.setattr(memory_module, "get_sync_memory_store", lambda: None)
    monkeypatch.setattr(agent_graph, "build_agent_graph_sync", lambda *args, **kwargs: expected_graph)

    assert openchatbi.get_default_graph() is expected_graph


@pytest.mark.asyncio
async def test_async_agent_graph_builds_with_langgraph_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_agent_graph_dependencies(monkeypatch)

    graph = await build_agent_graph_async(Mock(), checkpointer=None, memory_store=None, enable_context_management=False)

    assert graph.name == "agent_graph"
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream")


def test_text2sql_nested_invoke_keeps_dict_like_output() -> None:
    tool = get_sql_tools(FakeSQLGraph(), sync_mode=True)

    result = tool.invoke({"reasoning": "need SQL", "context": "show revenue"})

    assert "SQL Query" in result
    assert "SELECT 1" in result
    assert "Visualization Created: bar chart" in result


@pytest.mark.asyncio
async def test_text2sql_nested_ainvoke_keeps_dict_like_output() -> None:
    tool = get_sql_tools(FakeSQLGraph(), sync_mode=False)

    result = await tool.ainvoke({"reasoning": "need SQL", "context": "show revenue"})

    assert "SQL Query" in result
    assert "SELECT 1" in result
    assert "Visualization Created: bar chart" in result
