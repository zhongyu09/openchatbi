"""Unit tests for the out-of-band LLM-as-Judge evaluator."""

import json
import os
from unittest.mock import MagicMock

from evals.judge.llm_judge import JudgeVerdict, LLMAsJudgeEvaluator


def _evaluator_with_score(score, checks=None, reasons=None):
    from openchatbi.text2sql.confidence import ConfidenceResult

    inner = MagicMock()
    inner.evaluate.return_value = ConfidenceResult(
        score=score,
        reasons=reasons or ["looks correct"],
        checks=checks or {"select_columns": True, "where": True},
    )
    return LLMAsJudgeEvaluator(evaluator=inner, threshold=0.7)


def test_judge_passes_when_score_above_threshold():
    judge = _evaluator_with_score(0.9)
    verdict = judge.judge(
        question="How many orders?",
        generated_sql="SELECT COUNT(*) FROM orders",
        expected_sql="SELECT COUNT(*) FROM orders",
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.score == 0.9
    assert verdict.dimensions == {"select_columns": True, "where": True}


def test_judge_fails_when_score_below_threshold():
    judge = _evaluator_with_score(0.3)
    verdict = judge.judge(question="q", generated_sql="SELECT 1")
    assert verdict.passed is False
    assert "looks correct" in verdict.reasoning


def test_judge_passes_through_schema_and_expected_sql():
    inner = MagicMock()
    from openchatbi.text2sql.confidence import ConfidenceResult

    inner.evaluate.return_value = ConfidenceResult(score=0.8, reasons=[], checks={})
    judge = LLMAsJudgeEvaluator(evaluator=inner, threshold=0.7)
    judge.judge(question="q", generated_sql="SELECT 1", expected_sql="SELECT 2", schema={"t": ["c"]})
    _, kwargs = inner.evaluate.call_args
    assert kwargs["question"] == "q"
    assert kwargs["sql"] == "SELECT 1"
    assert kwargs["schema_info"] == {"t": ["c"]}
    # expected_sql is folded into the data_sample context the inner evaluator sees
    assert "SELECT 2" in (kwargs["data_sample"] or "")


# ---------------------------------------------------------------------------
# run_judge aggregation tests
# ---------------------------------------------------------------------------

from evals.judge import run_judge


def _write_case(tmp_path, name, category, sql):
    p = tmp_path / f"{name}.yaml"
    p.write_text(
        "id: %s\n"
        "category: %s\n"
        "input:\n"
        "  prompt: 'q for %s'\n"
        "gold:\n"
        "  expected_sql: \"%s\"\n"
        "  expected_tool_trajectory: ['text2sql']\n"
        "  expected_result_contains: ['x']\n" % (name, category, name, sql)
    )
    return p


def test_run_judge_aggregates_per_category(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "a1", "aggregation", "SELECT COUNT(*) FROM orders")
    _write_case(cases_dir, "j1", "join", "SELECT * FROM a JOIN b ON a.id=b.id")

    # Stub the judge so the test is hermetic (no LLM).
    from evals.judge.llm_judge import JudgeVerdict

    def fake_judge_factory():
        calls = {"n": 0}

        class _Stub:
            def judge(self, question, generated_sql, expected_sql=None, schema=None):
                calls["n"] += 1
                # aggregation passes, join fails
                passed = "JOIN" not in generated_sql
                return JudgeVerdict(
                    score=0.9 if passed else 0.3,
                    passed=passed,
                    reasoning="stub",
                    dimensions={},
                )

        return _Stub()

    monkeypatch.setattr(run_judge, "_build_judge", lambda: fake_judge_factory())
    out_path = tmp_path / "judge_out" / "report.json"
    exit_code = run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path))
    assert exit_code == 0
    report = json.loads(out_path.read_text())
    assert report["by_category"]["aggregation"]["pass_rate"] == 1.0
    assert report["by_category"]["join"]["pass_rate"] == 0.0
    assert report["overall"]["total"] == 2
    assert os.path.exists(out_path)
