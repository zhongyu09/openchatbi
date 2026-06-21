"""S2 — SQL quality evaluator (single structured-output LLM call, 6-step rubric).

Reused by HITL confidence gate, the Eval LLM-as-Judge, and the Memory
promotion gate. Default behaviour is OFF at every call site; this module only
computes a score when explicitly invoked.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass, field
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openchatbi.llm.llm import get_default_llm
from openchatbi.utils import log

# Ordered rubric check keys (Dataherald 6-step rubric).
RUBRIC_CHECKS: tuple[str, ...] = (
    "select_columns",  # SELECT columns map to the question's requested fields
    "where",  # WHERE conditions correctly express the filters
    "calc",  # calculations / aggregations are correct
    "subquery",  # subqueries are correctly decomposed
    "joins",  # JOIN columns match across tables
    "exec_result",  # the (sampled) execution result is plausible
)

_rubric_prompt_template_cache: str | None = None


def _get_rubric_prompt_template() -> str:
    """Get the rubric prompt template from prompts/sql_confidence_prompt.md with caching."""
    global _rubric_prompt_template_cache
    if _rubric_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("sql_confidence_prompt.md").open("r") as f:
            _rubric_prompt_template_cache = f.read()
    return _rubric_prompt_template_cache


@dataclass
class ConfidenceResult:
    score: float
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


class SimpleSQLEvaluator:
    """Single LLM call that scores SQL quality against a 6-step rubric."""

    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm if llm is not None else get_default_llm()

    def _low_temp_llm(self) -> BaseChatModel:
        # Bind a low temperature when the provider supports it; no-op otherwise.
        try:
            return cast(BaseChatModel, self.llm.bind(temperature=0.0))
        except Exception:
            return self.llm

    def evaluate(
        self,
        question: str,
        sql: str,
        schema_info: dict,
        data_sample: str | None,
        table_schema: str = "",
    ) -> ConfidenceResult:
        """Score the SQL against the 6-step rubric.

        Args:
            question: Natural-language question the SQL should answer.
            sql: SQL statement under review.
            schema_info: Result-set schema (columns/dtypes of the executed
                output); only reliable for the exec_result check.
            data_sample: Sampled rows of the execution result (may be None).
            table_schema: Source-table schema the SQL was written against —
                the reference for the structural checks 1-5 (select_columns,
                where, calc, subquery, joins).
        """
        prompt = (
            _get_rubric_prompt_template()
            .replace("[table_schema]", table_schema or "(not provided)")
            .replace("[result_schema]", json.dumps(schema_info, default=str))
            .replace("[data_sample]", data_sample or "")
            .replace("[sql]", sql)
            .replace("[question]", question)
        )
        messages = [
            SystemMessage(content="You are a precise SQL correctness evaluator."),
            HumanMessage(content=prompt),
        ]
        try:
            response = self._low_temp_llm().invoke(messages)
            return self._parse(getattr(response, "content", str(response)))
        except Exception as exc:  # never raise into the calling graph
            log(f"SimpleSQLEvaluator.evaluate failed: {exc}")
            return ConfidenceResult(score=0.0, reasons=[f"evaluator error: {exc}"], checks={})

    @staticmethod
    def _parse(content: str) -> ConfidenceResult:
        text = content.strip()
        # Tolerate fenced code blocks around the JSON payload.
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return ConfidenceResult(score=0.0, reasons=["unparseable evaluator output"], checks={})
        data = json.loads(text[start : end + 1])
        checks = {k: bool(data.get("checks", {}).get(k, False)) for k in RUBRIC_CHECKS}
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasons = [str(r) for r in data.get("reasons", [])]
        return ConfidenceResult(score=score, reasons=reasons, checks=checks)
