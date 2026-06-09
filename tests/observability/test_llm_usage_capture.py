"""Token/cost capture inside call_llm_chat_model_with_retry."""

from langchain_core.messages import AIMessage

from openchatbi.llm import llm as llm_mod
from openchatbi.observability import metrics
from openchatbi.observability.metrics import LLMCallRecord


class _UsageModel:
    """Minimal chat model returning a response with usage_metadata."""

    def invoke(self, messages, config=None):
        return AIMessage(
            content="SELECT 1",
            usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            response_metadata={"model_name": "gpt-4o"},
        )


def test_wrapper_records_usage(monkeypatch) -> None:
    captured: list[LLMCallRecord] = []
    monkeypatch.setattr(metrics, "record_llm_call", lambda rec: captured.append(rec))
    monkeypatch.setattr(llm_mod, "record_llm_call", metrics.record_llm_call, raising=False)

    resp = llm_mod.call_llm_chat_model_with_retry(
        _UsageModel(), [{"role": "user", "content": "hi"}], metadata={"node_name": "llm_node", "layer": "main"}
    )
    assert resp.content == "SELECT 1"
    assert len(captured) == 1
    rec = captured[0]
    assert rec.model == "gpt-4o"
    assert rec.input_tokens == 12
    assert rec.output_tokens == 4
    assert rec.total_tokens == 16
    assert rec.node == "llm_node"
    assert rec.layer == "main"
    assert rec.status == "success"
    assert rec.cost_usd > 0.0
