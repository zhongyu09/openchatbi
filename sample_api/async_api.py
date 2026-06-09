"""Async API for streaming chat responses from OpenChatBI."""

import asyncio
import dataclasses
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openchatbi import config
from openchatbi.agent_graph import build_agent_graph_async
from openchatbi.streaming import (
    AgentStreamProcessor,
    StreamInterrupt,
    StreamStep,
    StreamToken,
    extract_final_answer,
)
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
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
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
    # "events" → structured NDJSON (steps + tokens + final answer, like the
    # Streamlit UI). "text" → legacy plain-text answer-only stream.
    mode: str | None = "events"


def _event_to_dict(event) -> dict[str, Any]:
    """Serialize a streaming event into a JSON-friendly dict."""
    if isinstance(event, StreamStep):
        return {
            "type": "step",
            "kind": event.kind,
            "level": event.level,
            "label": event.label,
            "text": event.text,
            "data": _json_safe(event.data),
        }
    if isinstance(event, StreamToken):
        return {
            "type": "token",
            "level": event.level,
            "label": event.label,
            "is_final": event.is_final,
            "text": event.text,
        }
    if isinstance(event, StreamInterrupt):
        return {"type": "interrupt", "text": event.text, "buttons": _json_safe(event.buttons)}
    return {"type": "unknown"}


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion of arbitrary payloads to JSON-serializable data."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return str(obj)


@app.post("/chat/stream")
async def chat_stream(req: UserRequest):
    """Stream chat responses from the agent graph.

    When ``mode == "events"`` (default) the response is newline-delimited JSON
    (``application/x-ndjson``): one object per line with a ``type`` field of
    ``step`` | ``token`` | ``interrupt`` | ``final_answer``. This mirrors the
    intermediate steps the Streamlit UI displays. When ``mode == "text"`` the
    legacy plain-text answer-only stream is returned.
    """
    user_id = req.user_id or "default"
    session_id = req.session_id or "default"
    provider = req.provider

    # Create user-session ID just like in UI
    user_session_id = f"{user_id}-{session_id}"

    stream_input = {"messages": [("user", req.input)]}
    from openchatbi.observability.tracing import build_run_config

    config = build_run_config(user_id=user_id, session_id=session_id)

    try:
        graph = await get_or_build_graph(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def text_generator():
        """Legacy plain-text generator: yields only assistant answer text."""
        processor = AgentStreamProcessor()
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                if isinstance(event, StreamToken) and event.is_final and event.text:
                    yield event.text

    async def event_generator():
        """Structured NDJSON generator: steps, tokens, interrupts, final answer."""
        processor = AgentStreamProcessor()
        async for namespace, event_type, event_value in graph.astream(
            stream_input, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            for event in processor.process(namespace, event_type, event_value):
                yield json.dumps(_event_to_dict(event), ensure_ascii=False) + "\n"

        # Emit interrupt or final answer from the terminal state.
        state = await graph.aget_state(config)
        if state.interrupts:
            value = state.interrupts[0].value or {}
            interrupt = StreamInterrupt(text=value.get("text", ""), buttons=value.get("buttons", []) or [])
            yield json.dumps(_event_to_dict(interrupt), ensure_ascii=False) + "\n"
        else:
            final = extract_final_answer(processor.final_response)
            yield json.dumps({"type": "final_answer", "text": final}, ensure_ascii=False) + "\n"

    if (req.mode or "events") == "text":
        return StreamingResponse(text_generator(), media_type="text/plain")
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


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
