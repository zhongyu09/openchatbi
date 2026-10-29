"""Tests for tool-call validation and retry behavior in call_llm_chat_model_with_retry and agent_llm_call."""

from unittest.mock import Mock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from openchatbi.agent_graph import agent_llm_call
from openchatbi.graph_state import AgentState
from openchatbi.llm.llm import call_llm_chat_model_with_retry, has_malformed_tool_calls
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.search_knowledge import search_knowledge


@tool
def _dummy_text2sql(reasoning: str, context: str) -> str:
    """Fake text2sql tool for testing."""
    return "ok"


class TestCallLLMChatModelWithRetry:
    """Test call_llm_chat_model_with_retry's tool-name and malformed tool-call checks."""

    @pytest.mark.parametrize("chat_model_class", [ChatOpenAI, ChatAnthropic])
    def test_tool_call_accepted_without_bound_tools(self, chat_model_class):
        """Test that a tool call is accepted when bound_tools is omitted, in either bound-tool format."""
        llm = chat_model_class(model="test-model", api_key="test-key").bind_tools([search_knowledge, AskHuman])
        ai_message = AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {}, "id": "call_1"}])
        result_message = ChatResult(generations=[ChatGeneration(message=ai_message)])

        with patch.object(chat_model_class, "_generate", return_value=result_message) as generate:
            response = call_llm_chat_model_with_retry(llm, [HumanMessage(content="Show GMV by region")])

        assert generate.call_count == 1
        assert response.tool_calls[0]["name"] == "search_knowledge"

    def test_invalid_tool_calls_triggers_retry_and_recovers(self):
        """A response with non-empty invalid_tool_calls should feed the parse error back and retry."""
        bad = AIMessage(
            content="I found the open items. Let me get the status breakdown to summarize them properly.",
            invalid_tool_calls=[
                {
                    "name": "text2sql",
                    "args": '{"reasoning": "x", "context": Shared Context: ...}',
                    "id": "call_1",
                    "error": "Function text2sql arguments are not valid JSON",
                }
            ],
            response_metadata={"finish_reason": "tool_calls"},
        )
        good = AIMessage(
            content="",
            tool_calls=[{"name": "_dummy_text2sql", "args": {"reasoning": "x", "context": "y"}, "id": "call_2"}],
        )
        captured_calls: list[list] = []

        class RecordingFakeModel(GenericFakeChatModel):
            def invoke(self, input, config=None, **kwargs):
                captured_calls.append(list(input))
                return super().invoke(input, config=config, **kwargs)

        model = RecordingFakeModel(messages=iter([bad, good]))
        resp = call_llm_chat_model_with_retry(
            model,
            [HumanMessage("hi")],
            bound_tools=[_dummy_text2sql],
            parallel_tool_call=True,
        )

        assert len(captured_calls) == 2
        retry_prompt = captured_calls[1][-1]
        assert retry_prompt["role"] == "user"
        assert "not valid JSON" in retry_prompt["content"]
        assert "Function text2sql arguments are not valid JSON" in retry_prompt["content"]
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "_dummy_text2sql"
        assert resp.invalid_tool_calls == []

    def test_finish_reason_tool_calls_with_empty_tool_calls_triggers_retry(self):
        """A response with finish_reason=='tool_calls' but empty tool_calls is treated as malformed and retried."""
        empty_tool_call_msg = AIMessage(
            content="Let me run a query for that.",
            tool_calls=[],
            invalid_tool_calls=[],
            response_metadata={"finish_reason": "tool_calls"},
        )
        good = AIMessage(content="Final answer after retry.")

        model = GenericFakeChatModel(messages=iter([empty_tool_call_msg, good]))
        resp = call_llm_chat_model_with_retry(
            model,
            [HumanMessage("hi")],
            bound_tools=[_dummy_text2sql],
            parallel_tool_call=True,
        )

        assert resp.content == "Final answer after retry."
        assert not has_malformed_tool_calls(resp)

    def test_partial_invalid_tool_calls_in_parallel_mode_triggers_retry(self):
        """When parallel_tool_call=True and one tool call parses while another is invalid, retry the whole call."""
        mixed = AIMessage(
            content="Running queries...",
            tool_calls=[{"name": "_dummy_text2sql", "args": {"reasoning": "ok", "context": "ok"}, "id": "call_1"}],
            invalid_tool_calls=[
                {
                    "name": "_dummy_text2sql",
                    "args": "{bad json",
                    "id": "call_2",
                    "error": "Invalid JSON",
                }
            ],
        )
        good = AIMessage(
            content="",
            tool_calls=[{"name": "_dummy_text2sql", "args": {"reasoning": "ok", "context": "ok"}, "id": "call_3"}],
        )

        model = GenericFakeChatModel(messages=iter([mixed, good]))
        resp = call_llm_chat_model_with_retry(
            model,
            [HumanMessage("hi")],
            bound_tools=[_dummy_text2sql],
            parallel_tool_call=True,
        )

        assert resp.tool_calls[0]["id"] == "call_3"
        assert resp.invalid_tool_calls == []

    def test_exhausted_malformed_tool_calls_never_emitted_as_final_answer(self):
        """When all retries return invalid_tool_calls, _call_model returns an explicit error instead of transitional text."""
        transitional = "I found the open items. Let me get the status breakdown to summarize them properly."
        bad = AIMessage(
            content=transitional,
            invalid_tool_calls=[
                {
                    "name": "_dummy_text2sql",
                    "args": '{"reasoning": "x", "context": Shared Context: ...}',
                    "id": "call_1",
                    "error": "Function _dummy_text2sql arguments are not valid JSON",
                }
            ],
            response_metadata={"finish_reason": "tool_calls"},
        )

        call_count = 0

        class CountingFakeModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, input, config=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return super().invoke(input, config=config, **kwargs)

        model = CountingFakeModel(messages=iter([bad, bad, bad]))
        llm_node_fn = agent_llm_call(model, [_dummy_text2sql], context_manager=None)

        result = llm_node_fn(AgentState(messages=[HumanMessage(content="List open order items")]))

        assert call_count == 3
        assert result["agent_next_node"] == END
        assert transitional not in result["final_answer"]
        assert "the model produced an unusable tool call" in result["final_answer"]
        assert len(result["messages"]) == 1
        assert "the model produced an unusable tool call" in result["messages"][0].content
        assert result["messages"][0].invalid_tool_calls == []

    def test_finish_reason_tool_calls_without_tool_calls_not_emitted_as_final_answer(self):
        """Direct _call_model check: finish_reason=='tool_calls' with empty tool_calls produces explicit error."""
        transitional = "Let me check the schema first."
        bad = AIMessage(
            content=transitional,
            tool_calls=[],
            invalid_tool_calls=[],
            response_metadata={"finish_reason": "tool_calls"},
        )
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        with patch("openchatbi.agent_graph.call_llm_chat_model_with_retry", return_value=bad):
            llm_node_fn = agent_llm_call(mock_llm, [_dummy_text2sql], context_manager=None)
            result = llm_node_fn(AgentState(messages=[HumanMessage(content="hi")]))

        assert result["agent_next_node"] == END
        assert transitional not in result["final_answer"]
        assert "the model produced an unusable tool call" in result["final_answer"]
