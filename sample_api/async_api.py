"""Async API for streaming chat responses from OpenChatBI."""

import asyncio
from typing import Any
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from openchatbi import config
from openchatbi.agent_graph import build_agent_graph_async
from openchatbi.utils import get_report_download_response

# Session state storage: session_id -> state
sessions = defaultdict(dict)

# Graphs keyed by provider name
graphs: dict[str, Any] = {}
graphs_lock = asyncio.Lock()


async def get_or_build_graph(provider: str | None):
    """Get (or lazily build) a graph for the requested provider."""
    key = provider or "__default__"
    if key in graphs:
        return graphs[key]
    async with graphs_lock:
        if key in graphs:
            return graphs[key]
        graphs[key] = await build_agent_graph_async(config.get().catalog_store, llm_provider=provider)
        return graphs[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup: Initialize the async graph
    graphs["__default__"] = await build_agent_graph_async(config.get().catalog_store)
    yield
    # Shutdown: cleanup if needed
    graphs.clear()


app = FastAPI(lifespan=lifespan)


class UserRequest(BaseModel):
    """Request model for streaming chat."""

    input: str
    user_id: str | None = "default"
    session_id: str | None = "default"
    provider: str | None = None


@app.post("/chat/stream")
async def chat_stream(req: UserRequest):
    """Stream chat responses from the agent graph."""
    user_id = req.user_id or "default"
    session_id = req.session_id or "default"
    provider = req.provider

    # Create user-session ID just like in UI
    user_session_id = f"{user_id}-{session_id}"

    stream_input = {"messages": [("user", req.input)]}
    config = {"configurable": {"thread_id": user_session_id, "user_id": user_id}}

    try:
        graph = await get_or_build_graph(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def event_generator():
        """Generate streaming events from the graph."""
        async for _namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            text = ""
            if event_type == "messages":
                message_chunk = event_value[0]
                if isinstance(message_chunk, AIMessageChunk):
                    text = message_chunk.content
            elif event_value.get("llm_node") and event_value["llm_node"].get("final_answer"):
                text = event_value["llm_node"]["final_answer"]
            if text:
                yield text

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/user/{user_id}/memories")
async def get_user_memories(user_id: str):
    """Get all memories for a specific user."""
    try:
        # Import required modules for memory access
        import json

        from openchatbi.tool.memory import get_async_memory_store

        # Get the async memory store
        memory_store = await get_async_memory_store()

        memories = []
        namespace = ("memories", user_id)

        try:
            # Search for all memories for this user
            search_results = memory_store.search(namespace)

            for item in search_results:
                # Parse the memory data
                try:
                    content = json.loads(item.value.decode("utf-8")) if isinstance(item.value, bytes) else item.value
                except (json.JSONDecodeError, AttributeError):
                    content = str(item.value)

                memory_data = {
                    "key": item.key,
                    "content": content,
                    "namespace": str(namespace),
                    "created_at": getattr(item, "created_at", "Unknown"),
                    "updated_at": getattr(item, "updated_at", "Unknown"),
                }
                memories.append(memory_data)

            return {"user_id": user_id, "total_memories": len(memories), "memories": memories}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving memories: {str(e)}") from e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access memory store: {str(e)}") from e


@app.get("/api/download/report/{filename}")
async def download_report(filename: str):
    """Download a saved report file."""
    return get_report_download_response(filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
