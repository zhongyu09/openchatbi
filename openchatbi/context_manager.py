"""Context management utilities for handling long conversations."""

import json
import re
import uuid
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from openchatbi.context_config import ContextConfig, get_context_config
from openchatbi.llm.llm import call_llm_chat_model_with_retry
from openchatbi.prompts.system_prompt import get_summary_prompt_template
from openchatbi.utils import log


class ContextManager:
    """Manages conversation context to prevent token limit issues."""

    def __init__(self, llm: BaseChatModel, config: ContextConfig = None):
        """Initialize context manager.

        Args:
            llm: Language model for summarization
            config: Context configuration. If None, uses default config.
        """
        self.llm = llm
        self.config = config or get_context_config()

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    def manage_context_messages(self, messages: list) -> None:
        """Main context management function that directly modifies messages list.

        Args:
            messages: The list of messages to manage (modified in place)
        """
        if not self.config.enabled:
            return

        if not messages:
            return

        # Check if we need to manage context
        estimated_tokens = self.estimate_message_tokens(messages)
        if estimated_tokens <= self.config.summary_trigger_tokens:
            return  # No action needed

        log(f"Context management triggered: {estimated_tokens} tokens > {self.config.summary_trigger_tokens}")

        # Apply historical tool message compression directly
        self._compress_historical_tool_messages(messages)

        # Check if we still need summarization after compression
        remaining_tokens = self.estimate_message_tokens(messages)
        if remaining_tokens > self.config.summary_trigger_tokens and self.config.enable_summarization:
            self._apply_conversation_summarization(messages)

        log("Context management completed")

    # ============================================================================
    # TOKEN ESTIMATION METHODS
    # ============================================================================

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for most languages)."""
        return len(text) // 4

    def estimate_message_tokens(self, messages: List[BaseMessage]) -> int:
        """Estimate total tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(str(msg.content))
            # Add tokens for metadata and structure
            total += 50
        return total

    # ============================================================================
    # TOOL OUTPUT TRIMMING METHODS
    # ============================================================================

    def trim_tool_output(self, content: str, tool_name: str = "") -> str:
        """Trim tool output to manageable size while preserving key information."""
        if len(content) <= self.config.max_tool_output_length:
            return content

        # Preserve full error messages if configured
        if self.config.preserve_tool_errors and ("Error:" in content or "Traceback" in content):
            return content

        # For SQL results, preserve structure
        if "```sql" in content or "```csv" in content:
            return self._trim_structured_output(content)

        # For code execution results
        if "```python" in content or "Traceback" in content:
            return self._trim_code_output(content)

        # Generic trimming
        max_len = self.config.max_tool_output_length
        trimmed = content[: max_len // 2] + "\n\n... [Output truncated] ...\n\n" + content[-max_len // 2 :]
        return trimmed

    def _trim_structured_output(self, content: str) -> str:
        """Trim SQL/CSV output while preserving structure."""
        parts = []

        # Extract SQL query (always keep)
        sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
        if sql_match:
            parts.append(f"```sql\n{sql_match.group(1)}\n```")

        # Extract and trim CSV data
        csv_match = re.search(r"```csv\n(.*?)\n```", content, re.DOTALL)
        if csv_match:
            csv_data = csv_match.group(1)
            lines = csv_data.split("\n")
            max_rows = self.config.max_sql_result_rows

            if len(lines) > max_rows:  # Keep header + first half + last quarter
                keep_start = max_rows // 2
                keep_end = max_rows // 4
                trimmed_csv = "\n".join(
                    lines[: keep_start + 1]
                    + [f"... [{len(lines) - keep_start - keep_end - 1} rows omitted] ..."]
                    + lines[-keep_end:]
                )
                parts.append(f"```csv\n{trimmed_csv}\n```")
            else:
                parts.append(f"```csv\n{csv_data}\n```")

        # Keep visualization info
        viz_match = re.search(r"Visualization Created:.*", content)
        if viz_match:
            parts.append(viz_match.group(0))

        return "\n\n".join(parts)

    def _trim_code_output(self, content: str) -> str:
        """Trim Python code execution output."""
        # Keep error messages (full) if configured
        if self.config.preserve_tool_errors and ("Traceback" in content or "Error:" in content):
            return content

        lines = content.split("\n")
        max_lines = self.config.max_code_output_lines

        if len(lines) <= max_lines:
            return content

        # Keep first half and last quarter
        keep_start = max_lines // 2
        keep_end = max_lines // 4
        return "\n".join(lines[:keep_start] + ["... [Output truncated] ..."] + lines[-keep_end:])

    # ============================================================================
    # CONVERSATION SUMMARIZATION METHODS
    # ============================================================================

    def summarize_conversation(self, messages: List[BaseMessage]) -> str:
        """Create a summary of conversation history."""
        if not self.config.enable_conversation_summary:
            return ""

        # Filter out system messages for summarization
        # Note: The messages passed in are already historical messages (split point already calculated)
        messages_to_summarize = []
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                messages_to_summarize.append(msg)

        if not messages_to_summarize:
            return ""

        # Create summarization prompt
        conversation_text = self._format_messages_for_summary(messages_to_summarize)

        # Get the summary prompt template from the file and replace placeholder
        summary_prompt = get_summary_prompt_template().replace("[conversation_text]", conversation_text)

        try:
            response = call_llm_chat_model_with_retry(
                self.llm, [HumanMessage(content=summary_prompt)], parallel_tool_call=False
            )

            if isinstance(response, AIMessage):
                return f"[Conversation Summary]: {response.content}"
            return "[Summary generation failed]"

        except Exception as e:
            log(f"Failed to generate conversation summary: {e}")
            return "[Summary generation failed]"

    def _truncate_text(self, text: str, truncate_len: int = 500) -> str:
        # do not truncate Conversation Summary
        if text.startswith("[Conversation Summary]"):
            return text
        if len(text) > truncate_len:
            return text[:truncate_len] + "... [truncated]"
        return text

    def _format_messages_for_summary(self, messages: List[BaseMessage]) -> str:
        """Format messages for summary generation."""
        formatted = []
        max_messages = self.config.summary_max_messages

        # Limit messages for summary context
        for msg in messages[-max_messages:]:
            if isinstance(msg, HumanMessage):
                formatted.append(f"<user> {msg.content} </user>")
            elif isinstance(msg, AIMessage):
                content = msg.content or ""
                formatted.append(f"<assistant>")
                if isinstance(content, str):
                    formatted.append(self._truncate_text(content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            formatted.append(self._truncate_text(item))
                        elif isinstance(item, dict):
                            if item["type"] == "text":
                                formatted.append(self._truncate_text(item["text"]))
                            elif item["type"] == "tool_use":
                                formatted.append(json.dumps(item))
                formatted.append(f"</assistant>")
            elif isinstance(msg, ToolMessage):
                formatted.append(
                    f"<tool_result> tool_call_id: {msg.tool_call_id},  "
                    f"tool: {msg.name}, "
                    f"status: {msg.status}, "
                    f"result: {self._truncate_text(msg.content)} </tool_result>"
                )

        return "\n".join(formatted)

    # ============================================================================
    # CONTEXT MANAGEMENT IMPLEMENTATION METHODS
    # ============================================================================

    def _compress_historical_tool_messages(self, messages: List[BaseMessage]) -> None:
        """Compress historical (not recent) tool messages in place."""
        # Find a safe split point
        recent_start_index = self._find_safe_split_point(messages)

        # Find tool messages in historical part (before recent_start_index) that need compression
        for i in range(recent_start_index):
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                original_content = str(msg.content)

                # Apply intelligent filtering for tool message compression
                if self._should_compress_historical_tool_message(msg, original_content):
                    trimmed_content = self.trim_tool_output(original_content)

                    if len(trimmed_content) < len(original_content):
                        # Update message content directly
                        messages[i] = ToolMessage(
                            content=trimmed_content,
                            tool_call_id=msg.tool_call_id,
                            id=msg.id,  # Keep original ID to preserve position
                        )

                        log(
                            f"Compressed historical tool message: {len(original_content)} -> {len(trimmed_content)} chars"
                        )

    def _apply_conversation_summarization(self, messages: List[BaseMessage]) -> None:
        """Apply conversation summarization by modifying messages list in place."""
        if not self.config.enable_conversation_summary:
            return

        # Find a safe split point that doesn't separate AI messages with tool calls from their ToolMessages
        recent_start_index = self._find_safe_split_point(messages)

        if recent_start_index == 0:
            return  # No historical messages to summarize

        historical_messages = messages[:recent_start_index]
        recent_messages = messages[recent_start_index:]

        if len(historical_messages) == 1:
            msg = historical_messages[0]
            if isinstance(msg, AIMessage) and msg.content.startswith("[Conversation Summary]"):
                return

        # Generate summary
        summary_text = self.summarize_conversation(historical_messages)

        if summary_text:
            # Rebuild messages list in place: summary + recent
            new_messages = [AIMessage(content=summary_text, id=str(uuid.uuid4()))] + recent_messages

            # Clear and repopulate the list in place
            messages.clear()
            messages.extend(new_messages)

            log(f"Applied conversation summary, removed {len(historical_messages)} historical messages")

    def _find_safe_split_point(self, messages: List[BaseMessage]) -> int:
        """Find a safe split point that start at HumanMessage

        Returns the index where recent messages should start (everything before this index is historical).
        """
        if len(messages) <= self.config.keep_recent_messages:
            return 0  # Keep all messages as recent

        # If keep_recent_messages is 0, return all messages as historical
        if self.config.keep_recent_messages <= 0:
            return len(messages)

        # Start from the naive split point
        naive_split = len(messages) - self.config.keep_recent_messages

        # Find the nearest HumanMessage
        for i in range(naive_split, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage) or isinstance(msg, dict) and msg["role"] == "user":
                return i  # Split before this HumanMessage

        return naive_split

    # ============================================================================
    # CONTENT ANALYSIS HELPER METHODS
    # ============================================================================

    def _should_compress_historical_tool_message(self, tool_msg: ToolMessage, content: str) -> bool:
        """Determine if a historical tool message should be compressed.

        Args:
            tool_msg: The tool message to evaluate
            content: The content of the tool message

        Returns:
            bool: True if the message should be compressed
        """
        # Don't compress if content is already short
        if len(content) <= self.config.max_tool_output_length:
            return False

        # Always preserve error messages if configured
        if self.config.preserve_tool_errors and self._is_error_content(content):
            return False

        # Don't compress recent SQL results if configured
        if self.config.preserve_recent_sql and self._is_sql_content(content):
            return False

        # Compress large outputs from specific tools more aggressively
        if self._is_data_query_result(content):
            return True

        # Compress Python execution results but preserve errors
        if self._is_python_execution_result(content):
            return not self._is_error_content(content)

        # Default: compress if content is long
        return True

    def _is_error_content(self, content: str) -> bool:
        """Check if content contains error information."""
        error_indicators = [
            "error:",
            "Error:",
            "ERROR:",
            "exception:",
            "Exception:",
            "EXCEPTION:",
            "traceback",
            "Traceback",
            "TRACEBACK",
            "failed",
            "Failed",
            "FAILED",
            "KeyError",
            "ValueError",
            "TypeError",
            "AttributeError",
            "FileNotFoundError",
            "ConnectionError",
        ]
        return any(indicator in content for indicator in error_indicators)

    def _is_sql_content(self, content: str) -> bool:
        """Check if content contains SQL query results."""
        sql_indicators = [
            "```sql",
            "query results",
            "sql query:",
            "select ",
            "insert ",
            "update ",
            "delete ",
            "create table",
            "alter table",
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in sql_indicators)

    def _is_data_query_result(self, content: str) -> bool:
        """Check if content is a data query result that can be safely compressed."""
        indicators = [
            "```csv",
            "query results",
            "rows returned",
            "records found",
            "records in the database",
            "found records",
            "csv format",
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in indicators)

    def _is_python_execution_result(self, content: str) -> bool:
        """Check if content is Python code execution result."""
        indicators = [
            "```python",
            "execution completed",
            "output:",
            "result:",
            "print(",
        ]
        return any(indicator.lower() in content.lower() for indicator in indicators)
