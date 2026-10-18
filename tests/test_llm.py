"""Tests for the tool-name checks in call_llm_chat_model_with_retry."""

from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from openchatbi.llm.llm import call_llm_chat_model_with_retry
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.search_knowledge import search_knowledge


class TestCallLLMChatModelWithRetry:
    """Test call_llm_chat_model_with_retry's tool-name checks."""

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
