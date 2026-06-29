from __future__ import annotations

from dataclasses import dataclass, field

from openchatbi.text2sql.confidence import SimpleSQLEvaluator


@dataclass
class JudgeVerdict:
    score: float
    passed: bool
    reasoning: str
    dimensions: dict[str, bool] = field(default_factory=dict)


class LLMAsJudgeEvaluator:
    """Wraps the S2 SimpleSQLEvaluator to produce pass/fail verdicts for eval.

    Runs OUTSIDE RunLedger (custom assertion types are unsupported in runledger 0.1.1).
    """

    def __init__(self, evaluator: SimpleSQLEvaluator | None = None, threshold: float = 0.7) -> None:
        self._evaluator = evaluator or SimpleSQLEvaluator()
        self._threshold = threshold

    def judge(
        self,
        question: str,
        generated_sql: str,
        expected_sql: str | None = None,
        schema: dict | None = None,
        table_schema: str = "",
    ) -> JudgeVerdict:
        result = self._evaluator.evaluate(
            question=question,
            sql=generated_sql,
            schema_info=schema or {},
            data_sample=None,
            table_schema=table_schema,
            reference_sql=expected_sql,
        )
        return JudgeVerdict(
            score=result.score,
            passed=result.score >= self._threshold,
            reasoning="; ".join(result.reasons),
            dimensions=result.checks,
        )
