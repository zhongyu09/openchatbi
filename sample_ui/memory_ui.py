"""Memory listing UI for OpenChatBI using FastAPI and Gradio."""

import json
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI

from sample_ui.style import custom_css


def get_thread_memory_store() -> Any:
    """Create a thread-safe memory store connection."""
    try:
        import pysqlite3 as sqlite3
    except ImportError:
        import sqlite3
    from langgraph.store.sqlite import SqliteStore

    from openchatbi import config

    conn = sqlite3.connect("memory.db", check_same_thread=False)
    conn.isolation_level = None  # Use autocommit mode to avoid transaction conflicts
    store = SqliteStore(conn, index={"dims": 1536, "embed": config.get().embedding_model, "fields": ["text"]})
    try:
        store.setup()
    except Exception:
        pass  # Store might already be set up
    return store, conn


def list_all_memories() -> list[dict[str, Any]]:
    """
    Retrieve all memories from the memory store.

    Returns:
        List of memory items with their metadata
    """
    try:
        memory_store, conn = get_thread_memory_store()
        memories = []

        try:
            # Use search with partial namespace to find all memory items
            items = memory_store.search(("memories",), limit=1000)
            for item in items:
                memory_data = {
                    "namespace": item.namespace,
                    "key": item.key,
                    "value": item.value,
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)
        except Exception as e:
            return [{"error": f"Failed to retrieve memories: {str(e)}"}]
        finally:
            conn.close()

        return memories

    except Exception as e:
        return [{"error": f"Failed to access memory store: {str(e)}"}]


def format_memories_for_display(memories: list[dict[str, Any]]) -> str:
    """
    Format memories for display in the Gradio interface.

    Args:
        memories: List of memory items

    Returns:
        Formatted string for display
    """
    if not memories:
        return "No memories found."

    if len(memories) == 1 and "error" in memories[0]:
        return f"Error: {memories[0]['error']}"

    formatted = []
    for i, memory in enumerate(memories, 1):
        if "error" in memory:
            formatted.append(f"**Error:** {memory['error']}")
            continue

        formatted.append(f"## Memory {i}")
        formatted.append(f"**Namespace:** {memory['namespace']}")
        formatted.append(f"**Key:** {memory['key']}")

        # Format the value nicely
        value = memory["value"]
        if isinstance(value, dict):
            try:
                value_str = json.dumps(value, indent=2)
                formatted.append(f"**Content:**\n```json\n{value_str}\n```")
            except:
                formatted.append(f"**Content:** {str(value)}")
        else:
            formatted.append(f"**Content:** {str(value)}")

        formatted.append(f"**Created:** {memory['created_at']}")
        formatted.append(f"**Updated:** {memory['updated_at']}")
        formatted.append("---")

    return "\n".join(formatted)


def refresh_memories() -> list[list[str]]:
    """Refresh and return formatted memories."""
    memories = list_all_memories()
    return format_memories_for_display(memories)


def delete_memory_by_key(namespace_str: str, key: str) -> str:
    """
    Delete a memory by namespace and key.

    Args:
        namespace_str: String representation of namespace (e.g., "('memories', 'user1')")
        key: Memory key to delete

    Returns:
        Status message
    """
    try:
        import ast

        memory_store, conn = get_thread_memory_store()

        try:
            # Parse namespace string back to tuple
            namespace = ast.literal_eval(namespace_str)

            # Delete the item
            memory_store.delete(namespace, key)
            return f"Successfully deleted memory: {key} from namespace {namespace}"
        finally:
            conn.close()
    except Exception as e:
        return f"Failed to delete memory: {str(e)}"


# ---------- FastAPI ----------
app = FastAPI()

# ---------- Gradio UI ----------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Memory Store Viewer")
    gr.Markdown("View and manage long-term memories stored in the OpenChatBI system.")

    with gr.Row():
        with gr.Column(scale=3):
            memories_display = gr.Markdown(value=refresh_memories(), elem_id="memories-display")

        with gr.Column(scale=1):
            gr.Markdown("### Actions")
            refresh_btn = gr.Button("🔄 Refresh Memories", variant="primary")

            gr.Markdown("### Delete Memory")
            namespace_input = gr.Textbox(
                label="Namespace",
                placeholder="('memories', 'user_id')",
                info="Copy the exact namespace from the memory list",
            )
            key_input = gr.Textbox(
                label="Key", placeholder="memory_key", info="Copy the exact key from the memory list"
            )
            delete_btn = gr.Button("🗑️ Delete Memory", variant="stop")
            delete_status = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    refresh_btn.click(fn=refresh_memories, outputs=[memories_display])

    delete_btn.click(fn=delete_memory_by_key, inputs=[namespace_input, key_input], outputs=[delete_status]).then(
        fn=refresh_memories, outputs=[memories_display]
    )

# ---------- Application Startup ----------
# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/memory")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
