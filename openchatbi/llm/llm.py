import time
import traceback

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.base import RunnableBinding
from langchain_core.tools import StructuredTool

from openchatbi import config
from openchatbi.observability.metrics import LLMCallRecord, record_llm_call
from openchatbi.observability.pricing import estimate_cost
from openchatbi.tool.ask_human import AskHuman
from openchatbi.utils import log


def list_llm_providers() -> list[str]:
    """List configured LLM provider names (if any)."""
    try:
        providers = getattr(config.get(), "llm_providers", None) or {}
    except ValueError:
        return []
    return sorted(providers.keys())


def _get_provider_config(provider: str | None):
    cfg = config.get()
    providers = getattr(cfg, "llm_providers", None) or {}
    if not provider:
        provider = getattr(cfg, "llm_provider", None)
    if not provider:
        return None
    if provider not in providers:
        raise ValueError(f"Unknown llm_provider '{provider}'. Available: {sorted(providers.keys())}")
    return providers[provider]


def get_embedding_model(provider: str | None = None):
    """Get embedding model from config (optionally scoped to a provider)."""
    provider_cfg = _get_provider_config(provider)
    if provider_cfg and getattr(provider_cfg, "embedding_model", None) is not None:
        return provider_cfg.embedding_model
    return config.get().embedding_model


def get_default_llm(provider: str | None = None):
    """Get default LLM from config (optionally scoped to a provider)."""
    provider_cfg = _get_provider_config(provider)
    if provider_cfg:
        return provider_cfg.default_llm
    return config.get().default_llm


def get_llm(provider: str | None = None):
    """Get the chat model to use (alias for `get_default_llm`)."""
    return get_default_llm(provider)


def get_text2sql_llm(provider: str | None = None):
    """Get text2sql LLM from config (optionally scoped to a provider)."""
    provider_cfg = _get_provider_config(provider)
    if provider_cfg:
        return provider_cfg.text2sql_llm or provider_cfg.default_llm
    return config.get().text2sql_llm or get_default_llm()


def get_analysis_llm(provider: str | None = None):
    """Get the data analysis LLM from config (optionally scoped to a provider).

    Falls back to the default LLM when no dedicated `analysis_llm` is configured.
    """
    provider_cfg = _get_provider_config(provider)
    if provider_cfg:
        return provider_cfg.analysis_llm or provider_cfg.default_llm
    return config.get().analysis_llm or get_default_llm(provider)


def _invalid_tool_names(valid_tools, tool_calls) -> str:
    invalid_tools = []
    for tool in tool_calls:
        if tool["name"] not in valid_tools:
            invalid_tools.append(tool["name"])
    return ",".join(invalid_tools)


def call_llm_chat_model_with_retry(
    chat_model: BaseChatModel,
    messages,
    streaming_tokens=False,
    bound_tools=None,
    parallel_tool_call=False,
    metadata: dict | None = None,
):
    """Calls a language model chat endpoint with retry logic.

    Retries up to 3 times if there are errors or invalid tool calls.

    Args:
        chat_model: The chat model to invoke.
        messages (list): List of messages to send to the model.
        streaming_tokens (bool, optional): flag to indicate whether or not to show streaming tokens in UI.
        bound_tools (list, optional): List of valid tool names that can be called.
        parallel_tool_call (bool, optional): whether or not to call multiple tools in parallel.

    Returns:
        AIMessage or None: The model response or None if all retries failed.
    """
    new_messages = list(messages)
    valid_tools = []
    if bound_tools:
        for tool in bound_tools:
            if isinstance(tool, str):
                valid_tools.append(tool)
            elif isinstance(tool, StructuredTool):
                valid_tools.append(tool.name)
            elif tool == AskHuman:
                valid_tools.append("AskHuman")
    elif isinstance(chat_model, RunnableBinding) and "tools" in chat_model.kwargs:
        valid_tools += [tool["name"] for tool in chat_model.kwargs["tools"] if "name" in tool]
    extra_prompt = (
        " Please select the `AskHuman` tool if you need to confirm with user." if "AskHuman" in valid_tools else ""
    )
    response = None
    retry = 0
    # retry 3 times
    while retry < 3:
        start_time = time.time()
        try:
            log(f"Call LLM chat model with retry {retry} times.")
            response = chat_model.invoke(new_messages, config={"metadata": {"streaming_tokens": streaming_tokens}})
            elapsed = time.time() - start_time
            run_time = int(elapsed)
            log(f"LLM response after {run_time} seconds.")
            try:
                usage = getattr(response, "usage_metadata", None) or {}
                in_tok = int(usage.get("input_tokens", 0) or 0)
                out_tok = int(usage.get("output_tokens", 0) or 0)
                total_tok = int(usage.get("total_tokens", in_tok + out_tok) or (in_tok + out_tok))
                model_name = (getattr(response, "response_metadata", None) or {}).get("model_name", "") or ""
                meta = metadata or {}
                record_llm_call(
                    LLMCallRecord(
                        model=model_name,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=total_tok,
                        cost_usd=estimate_cost(model_name, in_tok, out_tok),
                        latency_s=elapsed,
                        node=meta.get("node_name"),
                        layer=meta.get("layer"),
                        status="success",
                    )
                )
            except Exception as exc:  # pragma: no cover - never break the call path
                log(f"LLM usage capture failed: {exc!r}")
        except Exception:
            run_time = int(time.time() - start_time)
            retry += 1
            log(f"LLM response error after {run_time} seconds, retry {retry} times.")
            log("===== Messages:")
            log(str(messages))
            traceback.print_exc()
            continue

        if response.tool_calls:
            if len(response.tool_calls) > 1 and not parallel_tool_call:
                retry += 1
                log(f"More than one tool {response.tool_calls}, retry {retry} times.")
                new_messages += [{"role": "user", "content": "You should only response with one tool call."}]
                response = None
                continue
            invalid_tools = _invalid_tool_names(valid_tools, response.tool_calls)
            if invalid_tools:
                retry += 1
                log(f"Invalid tool {invalid_tools}, retry {retry} times.")
                new_messages += [
                    {
                        "role": "user",
                        "content": f"You should not use tool that does not exist:`{invalid_tools}`."
                        f"Available tools are: {valid_tools}. Please choose a valid tool and try again."
                        f"{extra_prompt}",
                    }
                ]
                response = None
                continue
        break
    return response
