"""Tests for restoring chat history from the LangGraph checkpointer."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from sample_ui.history_loader import (
    load_session_history,
    load_session_history_tuples,
    messages_to_chat_tuples,
    messages_to_ui_history,
)


class TestMessagesToUiHistory:
    def test_empty_and_none_input(self):
        assert messages_to_ui_history(None) == []
        assert messages_to_ui_history([]) == []

    def test_human_message_mapping(self):
        result = messages_to_ui_history([HumanMessage(content="hello")])
        assert result == [{"role": "user", "type": "text", "content": "hello"}]

    def test_ai_text_message_mapping(self):
        result = messages_to_ui_history([AIMessage(content="hi there")])
        assert result == [{"role": "assistant", "type": "text", "content": "hi there"}]

    def test_skip_tool_and_system_messages(self):
        messages = [
            SystemMessage(content="you are a bot"),
            HumanMessage(content="q"),
            ToolMessage(content="tool output", tool_call_id="call_1"),
            AIMessage(content="a"),
        ]
        result = messages_to_ui_history(messages)
        assert result == [
            {"role": "user", "type": "text", "content": "q"},
            {"role": "assistant", "type": "text", "content": "a"},
        ]

    def test_skip_tool_call_only_ai_message(self):
        ai_tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
        )
        result = messages_to_ui_history([ai_tool_call])
        assert result == []

    def test_skip_empty_content(self):
        assert messages_to_ui_history([HumanMessage(content="   ")]) == []
        assert messages_to_ui_history([AIMessage(content="")]) == []

    def test_preserves_order(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="second"),
            HumanMessage(content="third"),
            AIMessage(content="fourth"),
        ]
        contents = [m["content"] for m in messages_to_ui_history(messages)]
        assert contents == ["first", "second", "third", "fourth"]

    def test_list_content_blocks(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "part1 "},
                {"type": "text", "text": "part2"},
                "trailing",
            ]
        )
        result = messages_to_ui_history([msg])
        assert result == [{"role": "assistant", "type": "text", "content": "part1 part2trailing"}]


class TestMessagesToChatTuples:
    def test_empty_and_none_input(self):
        assert messages_to_chat_tuples(None) == []
        assert messages_to_chat_tuples([]) == []

    def test_single_turn(self):
        result = messages_to_chat_tuples([HumanMessage(content="q"), AIMessage(content="a")])
        assert result == [("q", "a")]

    def test_multi_turn(self):
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        assert messages_to_chat_tuples(messages) == [("q1", "a1"), ("q2", "a2")]

    def test_skips_tool_and_system_messages(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="q"),
            ToolMessage(content="tool", tool_call_id="c1"),
            AIMessage(content="a"),
        ]
        assert messages_to_chat_tuples(messages) == [("q", "a")]

    def test_user_without_assistant(self):
        assert messages_to_chat_tuples([HumanMessage(content="q")]) == [("q", "")]

    def test_assistant_without_user(self):
        assert messages_to_chat_tuples([AIMessage(content="a")]) == [("", "a")]

    def test_consecutive_users(self):
        messages = [HumanMessage(content="q1"), HumanMessage(content="q2"), AIMessage(content="a")]
        assert messages_to_chat_tuples(messages) == [("q1", ""), ("q2", "a")]

    def test_consecutive_assistants(self):
        messages = [HumanMessage(content="q"), AIMessage(content="a1"), AIMessage(content="a2")]
        assert messages_to_chat_tuples(messages) == [("q", "a1"), ("", "a2")]


class _FakeGraph:
    def __init__(self, state=None, raise_exc=False):
        self._state = state
        self._raise = raise_exc

    async def aget_state(self, config):
        if self._raise:
            raise RuntimeError("boom")
        return self._state


def test_load_session_history_with_messages():
    state = SimpleNamespace(values={"messages": [HumanMessage(content="q"), AIMessage(content="a")]})
    graph = _FakeGraph(state=state)
    result = asyncio.run(load_session_history(graph, "user", "session"))
    assert result == [
        {"role": "user", "type": "text", "content": "q"},
        {"role": "assistant", "type": "text", "content": "a"},
    ]


def test_load_session_history_none_state():
    graph = _FakeGraph(state=None)
    assert asyncio.run(load_session_history(graph, "user", "session")) == []


def test_load_session_history_empty_values():
    graph = _FakeGraph(state=SimpleNamespace(values={}))
    assert asyncio.run(load_session_history(graph, "user", "session")) == []


def test_load_session_history_handles_exception():
    graph = _FakeGraph(raise_exc=True)
    assert asyncio.run(load_session_history(graph, "user", "session")) == []


def test_load_session_history_tuples_with_messages():
    state = SimpleNamespace(values={"messages": [HumanMessage(content="q"), AIMessage(content="a")]})
    graph = _FakeGraph(state=state)
    result = asyncio.run(load_session_history_tuples(graph, "user", "session"))
    assert result == [("q", "a")]


def test_load_session_history_tuples_none_state():
    graph = _FakeGraph(state=None)
    assert asyncio.run(load_session_history_tuples(graph, "user", "session")) == []


def test_load_session_history_tuples_handles_exception():
    graph = _FakeGraph(raise_exc=True)
    assert asyncio.run(load_session_history_tuples(graph, "user", "session")) == []
