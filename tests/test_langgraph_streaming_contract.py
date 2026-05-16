"""Streaming and interrupt/resume contract tests for LangGraph v1.1."""

from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


def _streaming_graph():
    def llm_node(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="hello")], "final_answer": "hello"}

    graph = StateGraph(MessagesState)
    graph.add_node("llm_node", llm_node)
    graph.add_edge(START, "llm_node")
    graph.add_edge("llm_node", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_default_astream_keeps_tuple_event_contract_for_gradio_flow() -> None:
    graph = _streaming_graph()

    events = [
        event
        async for event in graph.astream(
            {"messages": [("user", "hi")]},
            stream_mode=["updates", "messages"],
            subgraphs=True,
        )
    ]

    assert all(isinstance(event, tuple) and len(event) == 3 for event in events)
    assert {event_type for _namespace, event_type, _value in events} == {"messages", "updates"}

    message_event = next(event for event in events if event[1] == "messages")
    chunk, metadata = message_event[2]
    assert chunk.content == "hello"
    assert metadata["langgraph_node"] == "llm_node"


@pytest.mark.asyncio
async def test_streamlit_update_events_keep_node_payload_shape() -> None:
    graph = _streaming_graph()

    update_events = [
        event_value
        async for _namespace, event_type, event_value in graph.astream(
            {"messages": [("user", "hi")]},
            stream_mode=["updates", "messages"],
            subgraphs=True,
        )
        if event_type == "updates"
    ]

    assert update_events
    assert "llm_node" in update_events[0]
    assert update_events[0]["llm_node"]["messages"][0].content == "hello"


@pytest.mark.asyncio
async def test_first_visible_streaming_progress_arrives_within_two_seconds() -> None:
    graph = _streaming_graph()
    start = time.monotonic()

    async for _namespace, _event_type, _event_value in graph.astream(
        {"messages": [("user", "hi")]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        elapsed = time.monotonic() - start
        break
    else:  # pragma: no cover
        raise AssertionError("stream produced no visible progress")

    assert elapsed < 2.0


def test_interrupt_resume_state_contract() -> None:
    def ask_human(state: MessagesState) -> dict:
        user_feedback = interrupt({"text": "Need more info", "buttons": ["A", "B"]})
        return {"messages": [("human", user_feedback)]}

    graph = StateGraph(MessagesState)
    graph.add_node("ask_human", ask_human)
    graph.add_edge(START, "ask_human")
    graph.add_edge("ask_human", END)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-1", "user_id": "user-1"}}

    first = compiled.invoke({"messages": [("user", "hi")]}, config=config)
    state = compiled.get_state(config)

    assert "__interrupt__" in first
    assert state.interrupts
    assert state.interrupts[0].value["text"] == "Need more info"
    assert state.interrupts[0].value["buttons"] == ["A", "B"]

    resumed = compiled.invoke(Command(resume="answer"), config=config)

    assert resumed["messages"][-1].content == "answer"
    assert compiled.get_state(config).interrupts == ()


@pytest.mark.asyncio
async def test_visualization_update_event_shape_is_preserved() -> None:
    def generate_visualization(state: dict[str, Any]) -> dict:
        return {"visualization_dsl": {"chart_type": "bar"}}

    graph = StateGraph(dict)
    graph.add_node("generate_visualization", generate_visualization)
    graph.add_edge(START, "generate_visualization")
    graph.add_edge("generate_visualization", END)
    compiled = graph.compile()

    update_events = [
        value
        async for _namespace, event_type, value in compiled.astream(
            {"data": "value\n1"},
            stream_mode=["updates"],
            subgraphs=True,
        )
        if event_type == "updates"
    ]

    assert update_events[0]["generate_visualization"]["visualization_dsl"]["chart_type"] == "bar"
