"""Gradio-based Streaming UI for OpenChatBI with real-time chat interface."""

import asyncio
import sys
from collections import defaultdict
from contextlib import asynccontextmanager

import gradio as gr
import pysqlite3 as sqlite3
from fastapi import FastAPI
from langchain_core.messages import AIMessage

sys.modules["sqlite3"] = sqlite3

from langgraph.types import Command

from openchatbi.utils import get_report_download_response, get_text_from_message_chunk, log
from sample_ui.async_graph_manager import AsyncGraphManager
from sample_ui.plotly_utils import create_inline_chart_markdown, visualization_dsl_to_gradio_plot
from sample_ui.style import custom_css

# Session state storage: user_session_id -> state
session_interrupt = defaultdict(bool)

# Global event loop for async operations (similar to Streamlit approach)
global_event_loop = None


# Global graph manager (similar to Streamlit approach)
graph_manager = AsyncGraphManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for FastAPI lifespan"""
    # Startup
    await graph_manager.initialize()
    yield
    # Shutdown
    await graph_manager.cleanup()


# ---------- FastAPI ----------
app = FastAPI(lifespan=lifespan)


# ---------- Gradio UI functions ----------

def get_or_create_event_loop():
    """Get or create an independent event loop"""
    global global_event_loop

    if global_event_loop is None or global_event_loop.is_closed():
        global_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(global_event_loop)

    return global_event_loop


async def _async_respond_helper(message, chat_history, user_id, session_id):
    """
    Helper async function that contains the actual async logic.
    This will be run in an independent event loop.
    Collects all responses and returns them as a list.
    """
    responses = []  # Collect all yield values

    user_session_id = f"{user_id}-{session_id}"
    full_response = ""
    plot_figure = None
    chart_panel_update = gr.update()

    if session_interrupt[user_session_id]:
        stream_input = Command(resume=message)
    else:
        stream_input = {"messages": [{"role": "user", "content": message}]}

    config = {"configurable": {"thread_id": user_session_id, "user_id": user_id}}

    # Ensure graph is available
    if not graph_manager._initialized:
        try:
            await graph_manager.initialize()
        except Exception as e:
            log(f"Failed to initialize graph: {e}")
            chat_history[-1] = (chat_history[-1][0], f"Error: Failed to initialize system - {str(e)}")
            responses.append(("", chat_history, plot_figure, chart_panel_update))
            return responses

    data_csv = None
    # Asynchronously iterate through LangGraph stream
    async for _namespace, event_type, event_value in graph_manager.graph.astream(
        stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True, debug=True
    ):
        token = ""
        if event_type == "messages":
            chunk = event_value[0]
            metadata = event_value[1]
            # Keep llm node messages only to avoid duplicates
            if metadata["langgraph_node"] != "llm_node" or not metadata.get("streaming_tokens", False):
                continue
            token = get_text_from_message_chunk(chunk)
        else:
            # Process intermediate graph node updates
            if event_value.get("llm_node"):
                message_obj = event_value["llm_node"].get("messages")[0]
                if message_obj and isinstance(message_obj, AIMessage) and message_obj.tool_calls:
                    token = f"\nUse tool: {", ".join(tool["name"] for tool in message_obj.tool_calls)}\n"
                else:
                    token = "\n"
            elif event_value.get("information_extraction"):
                message_obj = event_value["information_extraction"].get("messages")[0]
                if message_obj.tool_calls:
                    token = f"Use tool: {message_obj.tool_calls[0]['name']}\n"
                else:
                    token = f"Rewrite question: {event_value['information_extraction'].get('rewrite_question')}\n"
            elif event_value.get("table_selection"):
                token = f"Selected tables: {event_value['table_selection'].get('tables')}\n"
            elif event_value.get("generate_sql"):
                token = f"SQL: \n ```sql \n{event_value['generate_sql'].get('sql')}\n```\n"
            elif event_value.get("execute_sql"):
                token = "Running SQL...\n"
                data_csv = event_value["execute_sql"].get("data")
            elif event_value.get("regenerate_sql"):
                token = f"SQL: \n ```sql \n{event_value['regenerate_sql'].get('sql')}\n```\n"
            elif event_value.get("generate_visualization"):
                visualization_dsl = event_value["generate_visualization"].get("visualization_dsl")
                # Check for visualization data in the final state and embed in response
                if visualization_dsl and "error" not in visualization_dsl and data_csv:
                    try:
                        plot_figure, plot_description = visualization_dsl_to_gradio_plot(data_csv, visualization_dsl)
                        # Add markdown representation to the chat
                        chart_markdown = create_inline_chart_markdown(data_csv, visualization_dsl)
                        full_response += f"\n\n{chart_markdown}"
                        chat_history[-1] = (chat_history[-1][0], full_response)
                        # Auto-show chart panel when plot is generated
                        chart_panel_update = gr.update(visible=True)
                        responses.append(("", chat_history, plot_figure, chart_panel_update))
                    except Exception as e:
                        log(f"Visualization generation error: {str(e)}")
                        full_response += f"\n\n⚠️ Visualization error: {str(e)}"
                        chat_history[-1] = (chat_history[-1][0], full_response)
                        responses.append(("", chat_history, plot_figure, chart_panel_update))

        # Update chat history with new tokens and collect response
        if token:
            full_response += token
            chat_history[-1] = (chat_history[-1][0], full_response)
            responses.append(("", chat_history, plot_figure, chart_panel_update))

    # Get final state and check for visualization data
    state = await graph_manager.graph.aget_state(config)
    final_state_values = state.values

    if state.interrupts:
        log(f"state.interrupts: {state.interrupts}")
        output_content = state.interrupts[0].value.get("text")
        if "buttons" in state.interrupts[0].value:
            output_content += str(state.interrupts[0].value.get("buttons"))
        full_response += output_content
        chat_history[-1] = (chat_history[-1][0], full_response)
        session_interrupt[user_session_id] = True
        responses.append(("", chat_history, plot_figure, chart_panel_update))
    else:
        session_interrupt[user_session_id] = False

    return responses


def respond(message, chat_history, user_id, session_id="default"):
    """
    Synchronous callback for Gradio Chatbot with streaming updates.

    This function processes user input and streams responses from the LangGraph agent.
    Returns: message_input, chat_history, plot_figure, chart_panel_visibility
    """
    # Add a placeholder in chat history
    chat_history.append((message, ""))
    plot_figure = None
    chart_panel_update = gr.update()
    yield "", chat_history, plot_figure, chart_panel_update  # Stream updates to UI

    # Get or create independent event loop
    loop = get_or_create_event_loop()

    # Run the async helper in the independent event loop
    try:
        responses = loop.run_until_complete(_async_respond_helper(message, chat_history, user_id, session_id))

        # Yield all collected responses
        for response in responses:
            yield response

    except Exception as e:
        log(f"Error in respond: {e}")
        import traceback

        traceback.print_exc()
        chat_history[-1] = (chat_history[-1][0], f"Error: {str(e)}")
        yield "", chat_history, plot_figure, chart_panel_update


# ---------- Memory Management Functions ----------
def list_user_memories(user_id: str) -> str:
    """List all memories for a specific user."""
    try:
        import json

        try:
            import pysqlite3 as sqlite3
        except ImportError:
            import sqlite3
        from langgraph.store.sqlite import SqliteStore

        from openchatbi import config

        # Create a new connection in this thread to avoid SQLite threading issues
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        conn.isolation_level = None  # Use autocommit mode to avoid transaction conflicts
        thread_memory_store = SqliteStore(
            conn, index={"dims": 1536, "embed": config.get().embedding_model, "fields": ["text"]}
        )
        try:
            thread_memory_store.setup()
        except Exception:
            pass  # Store might already be set up

        memories = []
        namespace = ("memories", user_id)

        try:
            # Use search with namespace to find all items for this user
            items = thread_memory_store.search(namespace, limit=1000)
            for item in items:
                memory_data = {
                    "key": item.key,
                    "value": item.value,
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)
        except Exception as e:
            return f"No memories found for user {user_id} or error: {str(e)}"
        finally:
            conn.close()

        if not memories:
            return f"No memories found for user {user_id}"

        formatted = [f"## Memories for User: {user_id}\n"]
        for i, memory in enumerate(memories, 1):
            formatted.append(f"### Memory {i}")
            formatted.append(f"**Key:** {memory['key']}")

            value = memory["value"]
            if isinstance(value, dict):
                try:
                    value_str = json.dumps(value, indent=2)
                    formatted.append(f"**Content:**\n```json\n{value_str}\n```")
                except ValueError:
                    formatted.append(f"**Content:** {str(value)}")
            else:
                formatted.append(f"**Content:** {str(value)}")

            formatted.append(f"**Created:** {memory['created_at']}")
            formatted.append(f"**Updated:** {memory['updated_at']}")
            formatted.append("---")

        return "\n".join(formatted)

    except Exception as e:
        return f"Error accessing memories: {str(e)}"


# ---------- Gradio UI Blocks ----------

# Create Gradio interface with custom CSS and theme
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 💬 OpenChatBI Agent Chatbot with Streaming & On-Demand Visualization")

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="Chat",
                        bubble_full_width=False,
                        height=500,
                        show_label=False,
                        sanitize_html=False,
                        render_markdown=True,
                    )
                    msg = gr.Textbox(placeholder="Type a message and press Enter", label="Input", elem_id="msg")

                with gr.Column(scale=2, visible=False) as chart_panel:
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### 📊 Interactive Chart")
                        with gr.Column(scale=1):
                            hide_chart_btn = gr.Button("✖️ Hide", elem_id="hide-chart-btn", size="sm")
                    plot = gr.Plot(label="", visible=True, show_label=False)

                with gr.Column(scale=1):
                    user_box = gr.Textbox(value="default", label="User ID", interactive=True)
                    session_box = gr.Textbox(value="default", label="Session ID", interactive=True)
                    show_chart_btn = gr.Button("📊 Show Chart Panel", variant="secondary")
                    gr.Markdown(
                        """
                    **Instructions**  
                    - Type a data question and press Enter
                    - Supports streaming output (real-time display)
                    - Click chart links in chat to view interactive charts
                    - Use 'Show Chart Panel' to make panel visible
                    - Session ID can be used to differentiate between conversations
                    """,
                        elem_id="description",
                    )

            def show_chart_panel():
                """Show the chart panel."""
                return gr.update(visible=True)

            def hide_chart_panel():
                """Hide the chart panel."""
                return gr.update(visible=False)

            # Register async submit handler for message input with plot output
            msg.submit(respond, [msg, chatbot, user_box, session_box], [msg, chatbot, plot, chart_panel])
            show_chart_btn.click(show_chart_panel, outputs=[chart_panel])
            hide_chart_btn.click(hide_chart_panel, outputs=[chart_panel])

        with gr.TabItem("🧠 Memory Store"):
            gr.Markdown("### Long-term Memory Viewer")
            gr.Markdown("View memories stored for each user in the system.")

            with gr.Row():
                with gr.Column(scale=3):
                    memory_display = gr.Markdown(
                        value="Enter a User ID and click 'Load Memories' to view stored memories.",
                        elem_id="memory-display",
                    )

                with gr.Column(scale=1):
                    memory_user_input = gr.Textbox(label="User ID", placeholder="default", value="default")
                    load_memories_btn = gr.Button("🔍 Load Memories", variant="primary")

            # Event handler for loading memories
            load_memories_btn.click(fn=list_user_memories, inputs=[memory_user_input], outputs=[memory_display])


# ---------- API Endpoints ----------
@app.get("/api/download/report/{filename}")
async def download_report(filename: str):
    """Download a saved report file."""
    return get_report_download_response(filename)


# ---------- Application Startup ----------
# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
