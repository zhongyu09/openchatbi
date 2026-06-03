"""Streamlit-based Streaming UI for OpenChatBI with collapsible thinking sections."""

import asyncio
import atexit
import sys
import traceback
import uuid
from pathlib import Path

import plotly.graph_objects as go

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover
    import sqlite3
import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

sys.modules["sqlite3"] = sqlite3

from langgraph.types import Command  # noqa: E402

from openchatbi import config as openchatbi_config  # noqa: E402
from openchatbi.llm.llm import list_llm_providers  # noqa: E402
from openchatbi.utils import get_text_from_message_chunk, log  # noqa: E402
from sample_ui.async_graph_manager import AsyncGraphManager  # noqa: E402
from sample_ui.plotly_utils import visualization_dsl_to_gradio_plot  # noqa: E402

# Configuration
st.set_page_config(page_title="OpenChatBI - Streamlit Interface", page_icon="💬", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = AsyncGraphManager()
if "session_interrupts" not in st.session_state:
    st.session_state.session_interrupts = {}
if "event_loop" not in st.session_state:
    st.session_state.event_loop = None

# Node names that belong to the text2sql SQL subgraph. Their intermediate
# token stream is intentionally not surfaced as raw "thinking" because the
# dedicated `updates` handlers below already render them (SQL, tables, etc.).
_SQL_SUBGRAPH_NODES = {
    "search_knowledge",
    "ask_human",
    "information_extraction",
    "table_selection",
    "generate_sql",
    "execute_sql",
    "regenerate_sql",
    "generate_visualization",
}

# Node names of the top-level agent graph (openchatbi). Anything else that
# bubbles up at depth 0 is the data analysis sub-agent: because it is invoked
# with a reset checkpoint namespace (see analysis/agent._build_sub_agent_config),
# the deepagents `model`/`tools`/middleware nodes are flattened onto depth 0
# instead of appearing as a nested subgraph.
_MAIN_GRAPH_NODES = {"llm_node", "use_tool", "ask_human"}

# Display label used for the (flattened) data analysis sub-agent layer.
_SUBAGENT_LABEL = "data_analysis"


def _format_namespace_label(namespace: tuple) -> str:
    """Turn an astream subgraph namespace into a readable layer label.

    A namespace looks like ``("use_tool:abc", "tools:xyz", ...)``; we keep the
    node-name part of each segment so users can see which layer a step belongs
    to (e.g. ``use_tool › tools``).
    """
    if not namespace:
        return "main"
    parts = []
    for seg in namespace:
        name = str(seg).split(":")[0]
        # Skip the bare numeric index layers LangGraph inserts for subgraphs
        # invoked without an explicit config (e.g. text2sql's sql_graph).
        if name.isdigit():
            continue
        parts.append(name)
    return " › ".join(parts) if parts else "subtask"


def _preview_text(text: object, limit: int = 300) -> str:
    """Collapse whitespace and truncate long content for compact previews."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + " …(truncated)"


def _describe_generic_node(node_output: dict) -> list[str]:
    """Describe tool calls / tool results from an arbitrary sub-agent node.

    Used as a fallback for nodes without a dedicated handler (notably the
    deepagents data-analysis sub-agent's ``model``/``tools`` nodes). Plain
    assistant text is skipped here because it is already shown via the token
    stream.
    """
    descriptions: list[str] = []
    for message in node_output.get("messages") or []:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(message, AIMessage) and tool_calls:
            for tool_call in tool_calls:
                args = tool_call.get("args") or {}
                # Show a short, human-readable rationale instead of dumping the
                # raw tool arguments (reasoning/context/code…) which read too
                # "techy". Fall back to just the tool name when none is present.
                rationale = ""
                for key in ("reasoning", "task", "question", "query", "goal"):
                    if args.get(key):
                        rationale = _preview_text(args[key], 200)
                        break
                suffix = f"：{rationale}" if rationale else ""
                descriptions.append(f"🛠️ Using tool: `{tool_call.get('name', '?')}`{suffix}")
        elif isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or "tool"
            descriptions.append(f"📤 Tool `{tool_name}` result：{_preview_text(message.content, 300)}")
    return descriptions


async def process_user_message_stream(
    message: str, user_id: str, session_id: str, llm_provider: str | None, thinking_container, response_container
):
    """
    Process user message through the OpenChatBI graph with real-time updates
    Updates the thinking_container and response_container as processing happens
    """
    thinking_steps = []
    final_response = ""
    plot_figure = None

    # Initialize graph if needed
    if not st.session_state.graph_manager._initialized:
        await st.session_state.graph_manager.initialize()
    graph = await st.session_state.graph_manager.get_graph(llm_provider)

    user_session_id = f"{user_id}-{session_id}"

    # Check for interrupts
    if st.session_state.session_interrupts.get(user_session_id, False):
        stream_input = Command(resume=message)
    else:
        stream_input = {"messages": [{"role": "user", "content": message}]}

    config = {"configurable": {"thread_id": user_session_id, "user_id": user_id}}

    data_csv = None

    # Use empty container for real-time updates
    thinking_placeholder = thinking_container.empty()

    # Build content chronologically - all events in time order
    base_content = "🔄 **Processing...**\n\n"
    chronological_content = ""  # All content in time order
    # Tracks which layer's tokens are currently being appended, so consecutive
    # tokens from the same source are grouped under one header. "main" = main
    # agent, a namespace tuple = a sub-agent, None = reset (after a step line).
    current_token_layer = None

    def update_display():
        full_content = base_content + chronological_content
        thinking_placeholder.markdown(full_content)

    # Initial display
    update_display()

    # Stream through the graph
    async for namespace, event_type, event_value in graph.astream(
        stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True, debug=True
    ):
        # Nesting depth from the subgraph namespace: 0 = main agent, >0 = a
        # genuinely nested subgraph (e.g. the text2sql SQL graph). Note the data
        # analysis sub-agent is flattened onto depth 0 (reset checkpoint ns).
        depth = len(namespace) if namespace else 0
        layer_label = _format_namespace_label(namespace)

        if event_type == "messages":
            chunk = event_value[0]
            metadata = event_value[1]
            node = metadata.get("langgraph_node")
            token = get_text_from_message_chunk(chunk)
            if not token:
                continue

            if depth == 0 and node == "llm_node" and metadata.get("streaming_tokens", False):
                # Main agent streaming its (intermediate/final) answer.
                final_response += token
                if current_token_layer != "main":
                    chronological_content += "\n\n**🤖 AI Response:** "
                    current_token_layer = "main"
                chronological_content += token
                update_display()
            elif node in _SQL_SUBGRAPH_NODES:
                # SQL subgraph internal LLM tokens: the dedicated `updates`
                # handlers below already render these as steps.
                continue
            elif depth == 0 and node in _MAIN_GRAPH_NODES:
                # Other main-graph nodes don't carry a useful thinking trace.
                continue
            else:
                # Sub-agent (data analysis) LLM thinking tokens. They bubble up
                # nested (depth>0) or flattened onto depth 0 (reset checkpoint ns).
                level = depth if depth > 0 else 1
                label = layer_label if depth > 0 else _SUBAGENT_LABEL
                token_layer = ("think", namespace, node)
                if current_token_layer != token_layer:
                    chronological_content += f"\n\n{'　' * level}💭 *{label} 思考:* "
                    current_token_layer = token_layer
                chronological_content += token
                update_display()

        else:
            # Process tool calls and intermediate steps. Works for the main
            # graph, the text2sql SQL subgraph, and the data analysis sub-agent.
            pending_steps = []  # list of (level, label, desc)

            for node_name, node_output in event_value.items():
                if not isinstance(node_output, dict):
                    continue

                # Decide which layer this node belongs to for display.
                is_main_node = depth == 0 and node_name in _MAIN_GRAPH_NODES
                if is_main_node:
                    level, label = 0, "main"
                elif depth == 0:
                    # deepagents sub-agent node flattened onto depth 0.
                    level, label = 1, _SUBAGENT_LABEL
                else:
                    level, label = depth, layer_label

                desc = None
                matched = True

                if node_name == "llm_node":
                    message_obj = (node_output.get("messages") or [None])[0]
                    if isinstance(message_obj, AIMessage) and message_obj.tool_calls:
                        sub_agents = [tool['name'] for tool in message_obj.tool_calls if
                                      tool['name'] in ('data_analysis', 'text2sql')]
                        normal_tools = [tool['name'] for tool in message_obj.tool_calls if
                                      tool['name'] not in sub_agents]
                        if normal_tools:
                            desc = f"🛠️ Using tools: {', '.join(normal_tools)}"
                        if sub_agents:
                            desc = f"🛠️ Running sub agent: {', '.join(sub_agents)}"

                elif node_name == "information_extraction":
                    message_obj = (node_output.get("messages") or [None])[0]
                    if message_obj and getattr(message_obj, "tool_calls", None):
                        desc = f"🛠️ Using tool: {message_obj.tool_calls[0]['name']}"
                    else:
                        rewrite_q = node_output.get("rewrite_question")
                        if rewrite_q:
                            desc = f"📝 Rewriting question: {rewrite_q}"

                elif node_name == "table_selection":
                    tables = node_output.get("tables")
                    if tables:
                        desc = f"🗂️ Selected tables: {tables}"

                elif node_name == "generate_sql":
                    sql = node_output.get("sql")
                    if sql:
                        desc = f"💾 Generated SQL:\n```sql\n{sql}\n```"

                elif node_name == "execute_sql":
                    desc = "⚡ Executing SQL query..."
                    data_csv = node_output.get("data")

                elif node_name == "regenerate_sql":
                    sql = node_output.get("sql")
                    if sql:
                        desc = f"🔄 Regenerated SQL:\n```sql\n{sql}\n```"

                elif node_name == "generate_visualization":
                    visualization_dsl = node_output.get("visualization_dsl")
                    if visualization_dsl and "error" not in visualization_dsl and data_csv:
                        try:
                            plot_figure, plot_description = visualization_dsl_to_gradio_plot(
                                data_csv, visualization_dsl
                            )
                            desc = f"📊 Generated visualization: {plot_description}"
                        except Exception as e:
                            desc = f"⚠️ Visualization error: {str(e)}"
                else:
                    matched = False

                if desc:
                    pending_steps.append((level, label, desc))

                # Generic fallback for sub-agent nodes (deepagents `model`/
                # `tools` etc.). They surface flattened on depth 0 (reset
                # checkpoint ns) or nested at depth>0 — never as a main node.
                if not matched and not is_main_node:
                    for generic_desc in _describe_generic_node(node_output):
                        pending_steps.append((level, label, generic_desc))

            for level, label, desc in pending_steps:
                thinking_steps.append(desc)
                step_number = len(thinking_steps)
                if chronological_content and not chronological_content.endswith("\n\n"):
                    chronological_content += "\n\n"
                if level == 0:
                    chronological_content += f"**Step {step_number}:** {desc}\n\n"
                else:
                    chronological_content += f"{'　' * level}↳ *[{label}]* {desc}\n\n"
                # Reset token grouping so the next streamed tokens get a header.
                current_token_layer = None
                update_display()

    # Check for interrupts in final state
    state = await graph.aget_state(config)
    if state.interrupts:
        log(f"State interrupts: {state.interrupts}")
        output_content = state.interrupts[0].value.get("text", "")
        if "buttons" in state.interrupts[0].value:
            output_content += str(state.interrupts[0].value.get("buttons"))
        final_response += output_content

        # Append interrupt content to chronological content
        chronological_content += output_content
        update_display()

        st.session_state.session_interrupts[user_session_id] = True
    else:
        st.session_state.session_interrupts[user_session_id] = False

    # Final update - add completion message to chronological content
    # Add some spacing if the last content didn't end with newlines
    if not chronological_content.endswith("\n\n"):
        chronological_content += "\n\n"
    chronological_content += "✅ **Analysis complete!**"
    update_display()

    # Extract final answer (last part without tool calls) and display outside thinking
    if final_response:
        # Find the last occurrence of tool usage to separate final answer
        lines = final_response.split("\n")
        final_answer_lines = []
        collecting_final = False

        for line in reversed(lines):
            if "Use tool:" in line or "Using tools:" in line or "Using tool:" in line:
                break
            final_answer_lines.append(line)
            collecting_final = True

        if collecting_final and final_answer_lines:
            # Reverse back to correct order
            final_answer_lines.reverse()
            final_answer_text = "\n".join(final_answer_lines).strip()

            if final_answer_text:
                with response_container:
                    processed_final_answer_text = process_download_links(final_answer_text)
                    render_content_with_downloads(processed_final_answer_text)

    # Final update to response container - only show plot if available (text response is in thinking container)
    with response_container:
        if plot_figure:
            st.plotly_chart(plot_figure, use_container_width=True, key=str(uuid.uuid4()))

    # Extract final answer for separate storage
    final_answer_text = ""
    if final_response:
        lines = final_response.split("\n")
        final_answer_lines = []
        collecting_final = False

        for line in reversed(lines):
            if "Use tool:" in line or "Using tools:" in line or "Using tool:" in line:
                break
            final_answer_lines.append(line)
            collecting_final = True

        if collecting_final and final_answer_lines:
            final_answer_lines.reverse()
            final_answer_text = "\n".join(final_answer_lines).strip()

    return final_response, plot_figure, thinking_steps, chronological_content, final_answer_text


def get_available_reports() -> list[str]:
    """Get list of available report files for download."""
    try:
        # Import config here to avoid circular imports
        from openchatbi import config

        report_dir = Path(config.get().report_directory)
        if not report_dir.exists():
            return []

        # Get all files in the report directory
        report_files = []
        for file_path in report_dir.iterdir():
            if file_path.is_file():
                report_files.append(file_path.name)

        return sorted(report_files)
    except Exception as e:
        st.error(f"Error accessing reports: {str(e)}")
        return []


def get_report_file_content(filename: str) -> tuple[bytes | None, str | None]:
    """Get report file content for download.

    Returns:
        tuple: (file_content_bytes, mime_type) or (None, None) if error
    """
    try:
        # Import config here to avoid circular imports
        from openchatbi import config

        report_dir = Path(config.get().report_directory)
        file_path = report_dir / filename

        # Security check - ensure file is within report directory
        if not file_path.exists() or not file_path.is_file():
            st.error(f"Report file not found: {filename}")
            return None, None

        try:
            file_path.resolve().relative_to(report_dir.resolve())
        except ValueError:
            st.error("Access denied to file")
            return None, None

        # Determine MIME type
        mime_type_map = {
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".html": "text/html",
            ".xml": "application/xml",
        }

        file_extension = file_path.suffix.lower()
        mime_type = mime_type_map.get(file_extension, "application/octet-stream")

        # Read file content
        with open(file_path, "rb") as f:
            content = f.read()

        return content, mime_type

    except Exception as e:
        st.error(f"Error reading report file: {str(e)}")
        return None, None


def process_download_links(content: str) -> str:
    """Process download links in content and replace them with Streamlit-compatible ones.

    Args:
        content: Message content that may contain download links

    Returns:
        str: Content with download links replaced
    """
    import re

    if not content:
        return content

    # Pattern to match both full URLs and path-only download links
    # Matches: http://localhost:8501/api/download/report/filename.ext or /api/download/report/filename.ext
    download_pattern = r"(?:https?://[^/\s]+)?/api/download/report/([^)\s\]<>]+)"

    def replace_link(match):
        filename = match.group(1)
        # Return a placeholder that we'll replace with actual download button
        return f"[DOWNLOAD_LINK:{filename}]"

    processed_content = re.sub(download_pattern, replace_link, content)

    # Debug log to see if processing worked
    if processed_content != content:
        st.write(f"🔍 Debug: Processed download links - found {content.count('/api/download/report/')} links")

    return processed_content


def render_content_with_downloads(content: str) -> None:
    """Render content and replace download placeholders with actual download buttons."""
    import re

    # Split content by download placeholders
    download_pattern = r"\[DOWNLOAD_LINK:([^)]+)\]"
    parts = re.split(download_pattern, content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular content
            if part.strip():
                st.markdown(part)
        else:
            # Download link filename
            filename = part
            file_content, mime_type = get_report_file_content(filename)

            if file_content is not None:
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=file_content,
                    file_name=filename,
                    mime=mime_type,
                    key=f"inline_download_{filename}_{hash(content)}",
                )
            else:
                st.error(f"❌ Could not load report: {filename}")


def display_message_with_thinking(
    role: str, content: str, thinking_steps: list[str] = None, plot_figure: go.Figure = None
):
    """Display a message with collapsible thinking section"""
    with st.chat_message(role):
        if thinking_steps and role == "assistant":
            # Create thinking section with all content inside
            with st.expander("💭 AI Thinking Process", expanded=False):
                for i, step in enumerate(thinking_steps, 1):
                    st.markdown(f"**Step {i}:** {step}")

                if content:
                    st.markdown("**🤖 AI Response:**")
                    render_content_with_downloads(content)

                st.success("✅ Analysis complete")

        # For non-assistant messages, display content normally
        elif content and role != "assistant":
            render_content_with_downloads(content)

        # Display plot if available (outside thinking container)
        if plot_figure:
            st.plotly_chart(plot_figure, use_container_width=True, key=str(uuid.uuid4()))


# Main UI
st.title("💬 OpenChatBI - Streamlit UI")
st.markdown("*AI-powered Business Intelligence Chat with Thinking*")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_id = st.text_input("User ID", value="default", help="Unique identifier for the user session")
    session_id = st.text_input("Session ID", value="default", help="Session identifier for conversation continuity")

    # Optional multi-provider support
    llm_provider = None
    provider_options = list_llm_providers()
    if provider_options:
        try:
            default_provider = getattr(openchatbi_config.get(), "llm_provider", None)
        except Exception:
            default_provider = None
        default_index = provider_options.index(default_provider) if default_provider in provider_options else 0
        llm_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=default_index,
            help="Select which configured LLM provider to use for this session",
        )

    st.markdown("---")
    st.markdown(
        """
    **💡 How to use:**
    - Type your business questions
    - Watch the AI thinking process in collapsible sections
    - View generated charts and analyses
    - Use different session IDs for separate conversations
    """
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Report Downloads")

    # Get available reports
    available_reports = get_available_reports()

    if available_reports:
        selected_report = st.selectbox(
            "Select a report to download:", options=[""] + available_reports, help="Choose a report file to download"
        )

        if selected_report and st.button("📥 Download Report"):
            file_content, mime_type = get_report_file_content(selected_report)
            if file_content is not None:
                st.download_button(
                    label=f"💾 Save {selected_report}",
                    data=file_content,
                    file_name=selected_report,
                    mime=mime_type,
                    key=f"download_{selected_report}",
                )
                st.success(f"✅ {selected_report} is ready for download!")
    else:
        st.info("No reports available for download.")

# Display chat history
for msg in st.session_state.messages:
    if msg["type"] == "chronological_message":
        # Display chronological content in expander - all collapsed after completion
        with st.chat_message(msg["role"]):
            with st.expander("💭 AI Thinking Process", expanded=False):
                st.markdown(msg["chronological_content"])

            # Extract and display final answer text outside thinking
            if msg.get("final_answer"):
                render_content_with_downloads(msg["final_answer"])

            # Display plot if available (outside thinking container)
            if msg.get("plot_figure"):
                st.plotly_chart(msg["plot_figure"], use_container_width=True, key=str(uuid.uuid4()))

    elif msg["type"] == "thinking_message":
        display_message_with_thinking(
            msg["role"], msg["content"], msg.get("thinking_steps", []), msg.get("plot_figure")
        )
    else:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                render_content_with_downloads(msg["content"])
            elif msg["type"] == "plot" and msg.get("plot_figure"):
                st.plotly_chart(msg["plot_figure"], use_container_width=True, key=str(uuid.uuid4()))

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response with real-time streaming
    with st.chat_message("assistant"):
        # Create thinking and response containers
        thinking_expander = st.expander("💭 AI Thinking Process...", expanded=True)
        thinking_container = thinking_expander.container()
        response_container = st.container()

        # Process the message asynchronously with real-time updates
        try:
            # Reuse the same event loop to avoid binding issues
            if st.session_state.event_loop is None or st.session_state.event_loop.is_closed():
                st.session_state.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(st.session_state.event_loop)

            loop = st.session_state.event_loop
            final_response, plot_figure, thinking_steps, full_chronological_content, final_answer = (
                loop.run_until_complete(
                    process_user_message_stream(
                        prompt, user_id, session_id, llm_provider, thinking_container, response_container
                    )
                )
            )

            # No need to create another expander - content is already shown in real-time
            # Process download links in the content before storing
            processed_chronological_content = process_download_links(full_chronological_content)
            processed_final_answer = process_download_links(final_answer) if final_answer else final_answer

            # Store the complete message with the processed content
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "type": "chronological_message",
                    "chronological_content": processed_chronological_content,
                    "final_answer": processed_final_answer,
                    "plot_figure": plot_figure,
                }
            )

            # Trigger rerun to collapse the thinking section
            st.rerun()

        except Exception as e:
            traceback.print_exc()
            st.error(f"❌ Error processing request: {str(e)}")
            error_content = f"❌ Error: {str(e)}"
            processed_error_content = process_download_links(error_content)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": processed_error_content})


# Cleanup on session end
def cleanup_session():
    """Cleanup resources when session ends"""
    if "graph_manager" in st.session_state:
        try:
            # Use the same event loop for cleanup
            if st.session_state.event_loop and not st.session_state.event_loop.is_closed():
                loop = st.session_state.event_loop
                loop.run_until_complete(st.session_state.graph_manager.cleanup())
                loop.close()
                st.session_state.event_loop = None
        except Exception as e:
            log(f"Error during session cleanup: {e}")


# Register cleanup (this is a simplified approach - in production you might want more robust cleanup)

atexit.register(cleanup_session)
