"""Tests for the S2 SQL quality evaluator (SimpleSQLEvaluator)."""

import json

import pytest
from langchain_core.language_models import FakeListChatModel

from openchatbi.text2sql.confidence import ConfidenceResult, SimpleSQLEvaluator


def test_confidence_result_fields():
    result = ConfidenceResult(
        score=0.83,
        reasons=["WHERE clause matches the date filter"],
        checks={
            "select_columns": True,
            "where": True,
            "calc": True,
            "subquery": True,
            "joins": True,
            "exec_result": True,
        },
    )
    assert result.score == 0.83
    assert result.reasons == ["WHERE clause matches the date filter"]
    assert result.checks["select_columns"] is True
    assert set(result.checks) == {
        "select_columns",
        "where",
        "calc",
        "subquery",
        "joins",
        "exec_result",
    }


def _verdict_json(score: float, all_true: bool = True) -> str:
    return json.dumps(
        {
            "score": score,
            "reasons": ["columns match", "where matches"],
            "checks": {
                "select_columns": all_true,
                "where": all_true,
                "calc": all_true,
                "subquery": all_true,
                "joins": all_true,
                "exec_result": all_true,
            },
        }
    )


def test_evaluate_parses_structured_verdict():
    mock_llm = FakeListChatModel(responses=[_verdict_json(0.92)])
    evaluator = SimpleSQLEvaluator(llm=mock_llm)
    result = evaluator.evaluate(
        question="How many users are there?",
        sql="SELECT COUNT(*) FROM users;",
        schema_info={"users": ["id", "name"]},
        data_sample="count\n42",
    )
    assert isinstance(result, ConfidenceResult)
    assert result.score == 0.92
    assert result.checks["joins"] is True
    assert all(result.checks[k] for k in result.checks)
    assert "columns match" in result.reasons


def test_evaluate_clamps_score_and_handles_false_checks():
    mock_llm = FakeListChatModel(responses=[_verdict_json(1.7, all_true=False)])
    evaluator = SimpleSQLEvaluator(llm=mock_llm)
    result = evaluator.evaluate("q", "SELECT 1", {}, None)
    assert result.score == 1.0  # clamped into [0, 1]
    assert result.checks["where"] is False


def test_evaluate_never_raises_on_bad_output():
    mock_llm = FakeListChatModel(responses=["not json at all"])
    evaluator = SimpleSQLEvaluator(llm=mock_llm)
    result = evaluator.evaluate("q", "SELECT 1", {}, None)
    assert result.score == 0.0
    assert result.checks == {}
