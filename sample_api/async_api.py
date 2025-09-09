"""Async API for streaming chat responses from OpenChatBI."""

from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from openchatbi.catalog.entry import catalog_store
from openchatbi.graph import build_agent_graph

# Session state storage: session_id -> state
sessions = defaultdict(dict)

graph = build_agent_graph(catalog_store, sync_mode=False)

app = FastAPI()


class UserRequest(BaseModel):
    """Request model for streaming chat."""

    input: str
    session_id: str | None = "default"


@app.post("/chat/stream")
async def chat_stream(req: UserRequest):
    """Stream chat responses from the agent graph."""
    session_id = req.session_id or "default"
    stream_input = {"messages": [("user", req.input)]}
    config = {"configurable": {"thread_id": session_id}}

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
            elif event_value.get("router") and event_value["router"].get("final_answer"):
                text = event_value["router"]["final_answer"]
            if text:
                yield text

    return StreamingResponse(event_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
