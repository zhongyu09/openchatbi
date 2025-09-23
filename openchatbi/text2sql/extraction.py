"""Information extraction module for text2sql processing."""

import traceback
from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from openchatbi.graph_state import SQLGraphState
from openchatbi.llm.llm import call_llm_chat_model_with_retry
from openchatbi.prompts.system_prompt import BASIC_KNOWLEDGE, EXTRACTION_PROMPT_TEMPLATE
from openchatbi.utils import extract_json_from_answer, get_text_from_content, log


def generate_extraction_prompt() -> str:
    """Generate extraction prompt.

    Returns:
        str: Generated prompt with placeholders replaced.
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE

    date_str = date.today().strftime("%Y-%m-%d")
    prompt = prompt.replace("[time_field_placeholder]", date_str)
    prompt = prompt.replace("[basic_knowledge_glossary]", BASIC_KNOWLEDGE)
    return prompt


def parse_extracted_info_json(llm_answer_content: Any) -> dict[str, Any]:
    """Extract and parse JSON from LLM response.

    Args:
        llm_answer_content: LLM response containing JSON.

    Returns:
        dict: Parsed JSON or empty dict if parsing fails.
    """
    try:
        text = get_text_from_content(llm_answer_content)
        result = extract_json_from_answer(text)
    except Exception:
        log(traceback.format_exc())
        result = {}
    return result


def information_extraction(llm: BaseChatModel) -> Callable:
    """Create function to extract information from questions.

    Args:
        llm (BaseChatModel): Language model for information extraction.

    Returns:
        function: Node function that extracts information from questions.
    """

    def _extract(state: SQLGraphState):
        """Extract information from question in state.

        Args:
            state (SQLGraphState): Current SQL graph state with question.

        Returns:
            dict: Updated state with extracted information.
        """
        messages = state["messages"]
        last_message = messages[-1]
        user_input = last_message.content
        log(f"information_extraction: {user_input}")
        system_prompt = generate_extraction_prompt()
        prompt = "Please extract the information according to the context."
        response = call_llm_chat_model_with_retry(
            llm, ([SystemMessage(system_prompt)] + messages + [HumanMessage(prompt)]), ["search_knowledge", "AskHuman"]
        )
        if response:
            log(response)
            if response.tool_calls:
                return {"messages": [response]}
            else:
                llm_answer_content = response.content
                parsed_result = parse_extracted_info_json(llm_answer_content)
                return {
                    "messages": [response],
                    "rewrite_question": parsed_result.get("rewrite_question"),
                    "info_entities": parsed_result,
                }
        else:
            return {"messages": [AIMessage(role="system", content="{}")]}

    return _extract


def information_extraction_conditional_edges(state: SQLGraphState):
    """Determine next node after information extraction.

    Args:
        state (SQLGraphState): Current SQL graph state.

    Returns:
        str: Next node ('ask_human', 'search_knowledge', 'next', or 'end').
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = None
    if isinstance(last_message, AIMessage):
        tool_calls = last_message.tool_calls
        log(f"tool_calls: {tool_calls}")
    if tool_calls:
        if tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        elif tool_calls[0]["name"] == "search_knowledge":
            return "search_knowledge"
        else:
            print(f"Unknown tool call: {tool_calls[0]['name']}")
            return "end"
    else:
        if "rewrite_question" in state:
            return "next"
        else:
            return "end"
