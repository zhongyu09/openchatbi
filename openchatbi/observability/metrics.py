"""LLM call metrics: an in-process record sink + optional Prometheus exporter."""

from __future__ import annotations

from dataclasses import dataclass

from openchatbi.utils import log


@dataclass
class LLMCallRecord:
    """One LLM invocation's accounting record."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    node: str | None
    layer: str | None
    status: str


# In-process ring of recent records (kept tiny; real sinks are Prometheus/trace).
_RECORDS: list[LLMCallRecord] = []
_MAX_RECORDS = 1000

# NOTE: generate_sql_node / regenerate_sql_node call ``llm.invoke()`` directly
# (generate_sql.py:315/448), bypassing call_llm_chat_model_with_retry. Their
# token usage is captured by the tracing callbacks registered in build_run_config
# (Task 4), NOT by routing them through this wrapper.


def record_llm_call(rec: LLMCallRecord) -> None:
    """Record an LLM call (best-effort; never raises into the call path)."""
    try:
        if rec is None:
            return
        _RECORDS.append(rec)
        if len(_RECORDS) > _MAX_RECORDS:
            del _RECORDS[0 : len(_RECORDS) - _MAX_RECORDS]
    except Exception as exc:  # pragma: no cover - defensive
        log(f"record_llm_call failed: {exc!r}")


def start_prometheus_exporter(port: int) -> None:
    """Start a Prometheus HTTP exporter if prometheus_client is installed (optional)."""
    try:
        from prometheus_client import start_http_server
    except ImportError:
        log("prometheus_client not installed; skipping exporter. Install openchatbi[observability].")
        return
    start_http_server(port)
