"""Tests for the S2 SQL quality evaluator (SimpleSQLEvaluator)."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

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


def _recording_llm() -> Mock:
    """LLM mock that records the prompt; bind() raises so invoke goes to the raw mock."""
    mock_llm = Mock()
    mock_llm.bind.side_effect = TypeError("no bind")
    mock_llm.invoke.return_value = AIMessage(content=_verdict_json(0.9))
    return mock_llm


def test_evaluate_prompts_with_source_schema_separated_from_result_schema():
    """Structural checks (1-5) must see the source-table schema; the result-set
    schema and data sample are scoped to the exec_result check only."""
    mock_llm = _recording_llm()
    evaluator = SimpleSQLEvaluator(llm=mock_llm)
    result = evaluator.evaluate(
        question="how many orders",
        sql="SELECT COUNT(*) AS cnt FROM Orders",
        schema_info={"columns": ["cnt"]},
        data_sample="cnt\n42",
        table_schema="## Table Orders\norder_id, customer_id, order_status",
    )
    assert result.score == 0.9
    prompt = mock_llm.invoke.call_args[0][0][1].content  # HumanMessage body
    assert "## Table Orders" in prompt
    assert '"columns": ["cnt"]' in prompt
    # Source schema section comes first and is labeled for checks 1-5;
    # result-set schema is explicitly scoped to check 6.
    assert prompt.index("Database schema") < prompt.index("Result-set schema")
    assert "ONLY for check 6" in prompt


def test_evaluate_table_schema_optional_for_backward_compat():
    mock_llm = _recording_llm()
    evaluator = SimpleSQLEvaluator(llm=mock_llm)
    evaluator.evaluate("q", "SELECT 1", {}, None)
    prompt = mock_llm.invoke.call_args[0][0][1].content
    assert "(not provided)" in prompt
