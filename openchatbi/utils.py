"""Utility functions for OpenChatBI."""

import json
import sys

from langchain_core.messages import AIMessageChunk
from regex import regex


def log(args) -> None:
    """Log messages to stderr for debugging."""
    print(args, file=sys.stderr, flush=True)


def get_text_from_content(content: str | list[str | dict]) -> str:
    """Extract text from various content formats.

    Args:
        content: String, list of strings, or list of dicts with 'text' key.

    Returns:
        str: Extracted text content.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        if isinstance(content[0], str):
            return "".join(content)
        elif isinstance(content[0], dict):
            return "".join([item.get("text", "") for item in content])
    return ""


def get_text_from_message_chunk(chunk: AIMessageChunk) -> str:
    """Extract content from an AIMessageChunk.

    Args:
        chunk (AIMessageChunk): The message chunk to extract text from.

    Returns:
        str: Extracted text content or empty string.
    """
    if not hasattr(chunk, "content") or not chunk.content:
        return ""
    return get_text_from_content(chunk.content)


def extract_json_from_answer(answer: str) -> dict:
    """Extract the first JSON object from a string answer.

    Args:
        answer (str): String that may contain JSON objects.

    Returns:
        dict: Parsed JSON object or empty dict if none found.
    """
    pattern = regex.compile(r"\{(?:[^{}]+|(?R))*\}")
    matches = pattern.findall(answer)
    json_result = matches[0] if matches else "{}"
    return json.loads(json_result)
