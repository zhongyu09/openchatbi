"""Tests for LLM call metrics recording."""

from openchatbi.observability import metrics
from openchatbi.observability.metrics import LLMCallRecord, record_llm_call


def test_record_llm_call_appends(monkeypatch) -> None:
    captured: list[LLMCallRecord] = []
    monkeypatch.setattr(metrics, "_RECORDS", captured, raising=False)
    rec = LLMCallRecord(
        model="gpt-4o",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=0.0001,
        latency_s=1.2,
        node="generate_sql",
        layer="text2sql",
        status="success",
    )
    record_llm_call(rec)
    assert captured == [rec]
    assert captured[0].total_tokens == 15
    assert captured[0].status == "success"


def test_record_llm_call_never_raises() -> None:
    # Recording must be best-effort: a malformed record must not propagate.
    record_llm_call(None)  # type: ignore[arg-type]
