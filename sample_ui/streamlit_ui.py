"""Streamlit-based Streaming UI for OpenChatBI with collapsible thinking sections."""

import asyncio
import sys
import traceback
import uuid
from typing import List

import plotly.graph_objects as go
import pysqlite3 as sqlite3
import streamlit as st

sys.modules["sqlite3"] = sqlite3

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from openchatbi.catalog.entry import catalog_store
from openchatbi.agent_graph import build_agent_graph_async
from openchatbi.llm.llm import default_llm
from openchatbi.tool.memory import cleanup_async_memory_store, get_async_memory_tools, setup_async_memory_store
from openchatbi.utils import get_text_from_message_chunk, log
from sample_ui.plotly_utils import visualization_dsl_to_gradio_plot

# Configuration
st.set_page_config(page_title="OpenChatBI - Streamlit Interface", page_icon="💬", layout="wide")


class AsyncGraphManager:
    """Manages the async graph and checkpointer lifecycle"""

    def __init__(self):
        self.checkpointer = None
        self.graph = None
        self._context_manager = None
        self._initialized = False

    async def initialize(self):
        """Initialize the graph and checkpointer"""
        if self._initialized:
            return

        try:
            # Setup async memory store
            await setup_async_memory_store()

            # Initialize checkpointer
            self._context_manager = AsyncSqliteSaver.from_conn_string("checkpoints.db")
            self.checkpointer = await self._context_manager.__aenter__()

            # Get async memory tools
            from openchatbi.tool.memory import get_async_memory_store

            async_store = await get_async_memory_store()
            async_memory_tools = await get_async_memory_tools(default_llm)

            # Build the graph
            self.graph = await build_agent_graph_async(
                catalog_store,
                checkpointer=self.checkpointer,
                memory_store=async_store,
                memory_tools=async_memory_tools,
            )

            self._initialized = True
            log("Graph initialized successfully")

        except Exception as e:
            log(f"Failed to initialize graph: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.checkpointer is not None and self._context_manager is not None:
            try:
                await self._context_manager.__aexit__(None, None, None)
                await cleanup_async_memory_store()
                log("Graph cleaned up successfully")
            except Exception as e:
                log(f"Error during cleanup: {e}")
            finally:
                self.checkpointer = None
                self.graph = None
                self._context_manager = None
                self._initialized = False


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = AsyncGraphManager()
if "session_interrupts" not in st.session_state:
    st.session_state.session_interrupts = {}
if "event_loop" not in st.session_state:
    st.session_state.event_loop = None


async def process_user_message_stream(
    message: str, user_id: str, session_id: str, thinking_container, response_container
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

    def update_display():
        full_content = base_content + chronological_content
        thinking_placeholder.markdown(full_content)

    # Initial display
    update_display()

    # Stream through the graph
    async for _namespace, event_type, event_value in st.session_state.graph_manager.graph.astream(
        stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True, debug=True
    ):
        if event_type == "messages":
            chunk = event_value[0]
            metadata = event_value[1]
            # Keep router node messages only to avoid duplicates
            if metadata["langgraph_node"] != "router":
                continue
            token = get_text_from_message_chunk(chunk)
            if token:
                final_response += token

                # Add to thinking content during processing
                if len(final_response) == len(token):
                    chronological_content += "\n**🤖 AI Response:** "
                chronological_content += token
                update_display()

        else:
            # Process tool calls and intermediate steps
            step_description = ""
            if event_value.get("router"):
                message_obj = event_value["router"].get("messages")[0]
                if message_obj and message_obj.tool_calls:
                    step_description = f"🛠️ Using tools: {', '.join(tool['name'] for tool in message_obj.tool_calls)}"

            elif event_value.get("information_extraction"):
                message_obj = event_value["information_extraction"].get("messages")[0]
                if message_obj and message_obj.tool_calls:
                    step_description = f"🛠️ Using tool: {message_obj.tool_calls[0]['name']}"
                else:
                    rewrite_q = event_value["information_extraction"].get("rewrite_question")
                    if rewrite_q:
                        step_description = f"📝 Rewriting question: {rewrite_q}"

            elif event_value.get("table_selection"):
                tables = event_value["table_selection"].get("tables")
                if tables:
                    step_description = f"🗂️ Selected tables: {tables}"

            elif event_value.get("generate_sql"):
                sql = event_value["generate_sql"].get("sql")
                if sql:
                    step_description = f"💾 Generated SQL:\n```sql\n{sql}\n```"

            elif event_value.get("execute_sql"):
                step_description = "⚡ Executing SQL query..."
                data_csv = event_value["execute_sql"].get("data")

            elif event_value.get("regenerate_sql"):
                sql = event_value["regenerate_sql"].get("sql")
                if sql:
                    step_description = f"🔄 Regenerated SQL:\n```sql\n{sql}\n```"

            elif event_value.get("generate_visualization"):
                visualization_dsl = event_value["generate_visualization"].get("visualization_dsl")
                if visualization_dsl and "error" not in visualization_dsl and data_csv:
                    try:
                        plot_figure, plot_description = visualization_dsl_to_gradio_plot(data_csv, visualization_dsl)
                        step_description = f"📊 Generated visualization: {plot_description}"
                    except Exception as e:
                        step_description = f"⚠️ Visualization error: {str(e)}"

            if step_description:
                thinking_steps.append(step_description)

                # Append new step to chronological content in time order
                step_number = len(thinking_steps)
                # Ensure proper spacing before new step
                if chronological_content and not chronological_content.endswith("\n\n"):
                    chronological_content += "\n\n"
                chronological_content += f"**Step {step_number}:** {step_description}\n\n"

                update_display()

    # Check for interrupts in final state
    state = await st.session_state.graph_manager.graph.aget_state(config)
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
                    st.markdown(final_answer_text)

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


def display_message_with_thinking(
    role: str, content: str, thinking_steps: List[str] = None, plot_figure: go.Figure = None
):
    """Display a message with collapsible thinking section"""
    with st.chat_message(role):
        if thinking_steps and role == "assistant":
            # Create thinking section with all content inside
            with st.expander("💭 AI Thinking Process", expanded=False):
                for i, step in enumerate(thinking_steps, 1):
                    st.markdown(f"**Step {i}:** {step}")

                if content:
                    st.markdown(f"**🤖 AI Response:** {content}")

                st.success("✅ Analysis complete")

        # For non-assistant messages, display content normally
        elif content and role != "assistant":
            st.markdown(content)

        # Display plot if available (outside thinking container)
        if plot_figure:
            st.plotly_chart(plot_figure, use_container_width=True, key=str(uuid.uuid4()))


# Main UI
st.title("💬 OpenChatBI - Streamlit Interface")
st.markdown("*AI-powered Business Intelligence Chat with Thinking*")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_id = st.text_input("User ID", value="default", help="Unique identifier for the user session")
    session_id = st.text_input("Session ID", value="default", help="Session identifier for conversation continuity")

    st.markdown("---")
    st.markdown(
        """
    **💡 How to use:**
    - Type your business intelligence questions
    - Watch the AI thinking process in collapsible sections
    - View generated charts and analyses
    - Use different session IDs for separate conversations
    """
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    if msg["type"] == "chronological_message":
        # Display chronological content in expander - all collapsed after completion
        with st.chat_message(msg["role"]):
            with st.expander("💭 AI Thinking Process", expanded=False):
                st.markdown(msg["chronological_content"])

            # Extract and display final answer text outside thinking
            if msg.get("final_answer"):
                st.markdown(msg["final_answer"])

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
                st.markdown(msg["content"])
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
                    process_user_message_stream(prompt, user_id, session_id, thinking_container, response_container)
                )
            )

            # No need to create another expander - content is already shown in real-time
            # Store the complete message with the full chronological content
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "type": "chronological_message",
                    "chronological_content": full_chronological_content,
                    "final_answer": final_answer,
                    "plot_figure": plot_figure,
                }
            )

            # Trigger rerun to collapse the thinking section
            st.rerun()

        except Exception as e:
            traceback.print_exc()
            st.error(f"❌ Error processing request: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"❌ Error: {str(e)}"})


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
import atexit

atexit.register(cleanup_session)
