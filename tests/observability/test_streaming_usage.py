"""AgentStreamProcessor aggregates per-turn token/cost into StreamUsage."""

from langchain_core.messages import AIMessageChunk

from openchatbi.streaming import AgentStreamProcessor, StreamUsage


def _msg_event(chunk: AIMessageChunk, node: str) -> tuple:
    # Mirrors astream(stream_mode=["messages"]) triple: (namespace, "messages", (chunk, metadata)).
    return ((), "messages", (chunk, {"langgraph_node": node, "streaming_tokens": True}))


def test_turn_usage_accumulates_from_final_chunk() -> None:
    processor = AgentStreamProcessor()
    chunk = AIMessageChunk(
        content="answer",
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        response_metadata={"model_name": "gpt-4o"},
    )
    events = processor.process(*_msg_event(chunk, "llm_node"))
    # The token still streams; usage is folded into the accumulator.
    assert processor.turn_usage.turn_tokens == 120
    assert processor.turn_usage.by_model.get("gpt-4o") == 120
    assert processor.turn_usage.turn_cost_usd > 0.0

    usage_event = processor.emit_turn_usage()
    assert isinstance(usage_event, StreamUsage)
    assert usage_event.turn_tokens == 120


def test_emit_turn_usage_none_when_no_usage() -> None:
    processor = AgentStreamProcessor()
    assert processor.emit_turn_usage() is None
