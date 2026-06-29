"""Unit tests for the out-of-band LLM-as-Judge evaluator."""

import json
import os
from unittest.mock import MagicMock

from evals.judge import run_judge
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
    # expected_sql is passed as a reference solution, not an execution sample.
    assert kwargs["reference_sql"] == "SELECT 2"
    assert kwargs["data_sample"] is None


# ---------------------------------------------------------------------------
# run_judge aggregation tests
# ---------------------------------------------------------------------------


def _write_case(tmp_path, name, category, sql):
    p = tmp_path / f"{name}.yaml"
    p.write_text(
        f"id: {name}\n"
        f"category: {category}\n"
        "input:\n"
        f"  prompt: 'q for {name}'\n"
        "gold:\n"
        f'  expected_sql: "{sql}"\n'
        "  expected_tool_trajectory: ['text2sql']\n"
        "  expected_result_contains: ['x']\n"
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


def test_run_judge_writes_incremental_report_before_later_failure(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "aggregation", "SELECT COUNT(*) FROM orders")
    _write_case(cases_dir, "c02", "join", "SELECT * FROM a JOIN b ON a.id=b.id")

    class _FailOnSecondJudge:
        def __init__(self):
            self.calls = 0

        def judge(self, question, generated_sql, expected_sql=None, schema=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("judge failed")
            return JudgeVerdict(score=0.9, passed=True, reasoning="stub", dimensions={})

    monkeypatch.setattr(run_judge, "_build_judge", lambda: _FailOnSecondJudge())
    out_path = tmp_path / "judge_out" / "report.json"

    try:
        run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path))
    except RuntimeError as exc:
        assert str(exc) == "judge failed"
    else:
        raise AssertionError("expected judge failure")

    report = json.loads(out_path.read_text())
    assert report["progress"] == {"processed": 1, "total": 2, "complete": False}
    assert report["overall"]["total"] == 1
    assert [case["id"] for case in report["cases"]] == ["c01"]


# ---------------------------------------------------------------------------
# New tests: --generated path and smoke mode
# ---------------------------------------------------------------------------


def _make_stub_judge(received_calls: list):
    """Return a fake judge factory that records call args."""
    from evals.judge.llm_judge import JudgeVerdict

    class _Stub:
        def judge(self, question, generated_sql, expected_sql=None, schema=None):
            received_calls.append(
                {
                    "question": question,
                    "generated_sql": generated_sql,
                    "expected_sql": expected_sql,
                }
            )
            return JudgeVerdict(score=0.9, passed=True, reasoning="stub", dimensions={})

    return _Stub()


def test_generated_map_json_passes_agent_sql_not_gold(tmp_path, monkeypatch):
    """When --generated is supplied the judge receives the AGENT's sql, not the gold."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    gold_sql = "SELECT COUNT(*) FROM orders"
    agent_sql = "SELECT count(id) FROM orders"
    _write_case(cases_dir, "c01", "aggregation", gold_sql)

    # Write a simple JSON map keyed by case id.
    gen_map = tmp_path / "generated.json"
    gen_map.write_text(json.dumps({"c01": agent_sql}))

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    rc = run_judge.run(
        cases_dir=str(cases_dir),
        out_path=str(out_path),
        generated_path=str(gen_map),
    )
    assert rc == 0
    assert len(calls) == 1
    # The evaluator must receive the AGENT SQL, not the gold SQL.
    assert calls[0]["generated_sql"] == agent_sql
    assert calls[0]["expected_sql"] == gold_sql

    report = json.loads(out_path.read_text())
    assert report["mode"] == "generated"
    assert report["overall"]["skipped"] == 0
    assert report["overall"]["evaluated"] == 1
    assert report["overall"]["pass_rate"] == 1.0


def test_generated_map_jsonl_passes_agent_sql(tmp_path, monkeypatch):
    """JSONL format is also accepted."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    gold_sql = "SELECT * FROM a JOIN b ON a.id=b.id"
    agent_sql = "SELECT a.col FROM a INNER JOIN b ON a.id = b.id"
    _write_case(cases_dir, "j1", "join", gold_sql)

    gen_jsonl = tmp_path / "generated.jsonl"
    gen_jsonl.write_text(json.dumps({"id": "j1", "prompt": "q for j1", "generated_sql": agent_sql}) + "\n")

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    rc = run_judge.run(
        cases_dir=str(cases_dir),
        out_path=str(out_path),
        generated_path=str(gen_jsonl),
    )
    assert rc == 0
    assert calls[0]["generated_sql"] == agent_sql


def test_generated_map_prompt_fallback(tmp_path, monkeypatch):
    """When the map key is the prompt (not the case id), lookup still succeeds."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    gold_sql = "SELECT COUNT(*) FROM orders"
    agent_sql = "SELECT count(*) FROM orders WHERE 1=1"
    _write_case(cases_dir, "c01", "aggregation", gold_sql)

    # Key by prompt string, not case id.
    gen_map = tmp_path / "generated.json"
    gen_map.write_text(json.dumps({"q for c01": agent_sql}))

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    run_judge.run(
        cases_dir=str(cases_dir),
        out_path=str(out_path),
        generated_path=str(gen_map),
    )
    assert calls[0]["generated_sql"] == agent_sql


def test_missing_case_in_generated_map_is_skipped(tmp_path, monkeypatch):
    """Cases absent from the generated map appear as skipped, excluded from pass_rate."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "aggregation", "SELECT COUNT(*) FROM orders")
    _write_case(cases_dir, "c02", "aggregation", "SELECT SUM(total) FROM orders")

    # Only c01 has a generated entry; c02 is missing.
    gen_map = tmp_path / "generated.json"
    gen_map.write_text(json.dumps({"c01": "SELECT count(*) FROM orders"}))

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    rc = run_judge.run(
        cases_dir=str(cases_dir),
        out_path=str(out_path),
        generated_path=str(gen_map),
    )
    assert rc == 0
    # Judge called only once (the matched case).
    assert len(calls) == 1

    report = json.loads(out_path.read_text())
    assert report["overall"]["total"] == 2
    assert report["overall"]["evaluated"] == 1
    assert report["overall"]["skipped"] == 1
    # pass_rate is over evaluated only.
    assert report["overall"]["pass_rate"] == 1.0

    # Find the skipped case in cases list.
    skipped = [c for c in report["cases"] if c["skipped"]]
    assert len(skipped) == 1
    assert skipped[0]["id"] == "c02"
    assert skipped[0].get("skip_reason") == "no_generated_sql"

    # Category aggregation also tracks skipped.
    agg = report["by_category"]["aggregation"]
    assert agg["skipped"] == 1
    assert agg["evaluated"] == 1
    assert agg["pass_rate"] == 1.0


def test_smoke_mode_default(tmp_path, monkeypatch, capsys):
    """When no --generated is passed, smoke mode runs gold-vs-gold and warns on stderr."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    gold_sql = "SELECT COUNT(*) FROM orders"
    _write_case(cases_dir, "c01", "aggregation", gold_sql)

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    rc = run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path), generated_path=None)
    assert rc == 0

    # Mode field present.
    report = json.loads(out_path.read_text())
    assert report["mode"] == "smoke"

    # In smoke mode the judge receives gold SQL as both generated and expected.
    assert calls[0]["generated_sql"] == gold_sql
    assert calls[0]["expected_sql"] == gold_sql

    # Warning emitted to stderr.
    captured = capsys.readouterr()
    assert "SMOKE" in captured.err
    assert "--generated" in captured.err


def test_run_judge_emits_progress_logs(tmp_path, monkeypatch, capsys):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "aggregation", "SELECT COUNT(*) FROM orders")

    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge([]))
    out_path = tmp_path / "report.json"

    rc = run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path), generated_path=None)

    assert rc == 0
    log = capsys.readouterr().err
    assert "=== Judge Evaluation Plan ===" in log
    assert "Mode: smoke" in log
    assert "JUDGING CASE 1/1: c01" in log
    assert "judged 1/1 cases (passed, score=0.900)" in log
    assert "Judge evaluation complete: 1/1 cases" in log


def test_pass_rate_over_evaluated_only(tmp_path, monkeypatch):
    """pass_rate denominator excludes skipped cases."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    # 3 cases; only 2 matched; 1 passes, 1 fails → pass_rate should be 0.5, not 0.333.
    _write_case(cases_dir, "c01", "cat", "SELECT 1")
    _write_case(cases_dir, "c02", "cat", "SELECT 2")
    _write_case(cases_dir, "c03", "cat", "SELECT 3")

    gen_map = tmp_path / "generated.json"
    gen_map.write_text(json.dumps({"c01": "SELECT 1 ok", "c02": "SELECT 2 ok"}))

    from evals.judge.llm_judge import JudgeVerdict

    n = {"i": 0}

    class _AltStub:
        def judge(self, question, generated_sql, expected_sql=None, schema=None):
            n["i"] += 1
            # first call passes, second fails
            passed = n["i"] == 1
            return JudgeVerdict(score=0.9 if passed else 0.3, passed=passed, reasoning="s", dimensions={})

    monkeypatch.setattr(run_judge, "_build_judge", lambda: _AltStub())
    out_path = tmp_path / "report.json"
    run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path), generated_path=str(gen_map))

    report = json.loads(out_path.read_text())
    assert report["overall"]["evaluated"] == 2
    assert report["overall"]["skipped"] == 1
    assert report["overall"]["passed"] == 1
    assert report["overall"]["pass_rate"] == 0.5  # 1/2, not 1/3


# ---------------------------------------------------------------------------
# Integration-style test with real case files but mocked evaluator
# ---------------------------------------------------------------------------


def test_config_arg_triggers_config_load(tmp_path, monkeypatch):
    """--config makes run() reload openchatbi config so the judge uses that LLM."""
    import openchatbi

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "aggregation", "SELECT COUNT(*) FROM orders")

    loaded: list = []
    monkeypatch.setattr(openchatbi.config, "load", lambda p: loaded.append(p))
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge([]))

    out_path = tmp_path / "report.json"
    run_judge.run(
        cases_dir=str(cases_dir),
        out_path=str(out_path),
        generated_path=None,
        config_path="my_config.yaml",
    )
    assert loaded == ["my_config.yaml"]


def test_no_config_arg_does_not_reload(tmp_path, monkeypatch):
    """Without --config, run() must NOT call config.load (preserves old behavior)."""
    import openchatbi

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "aggregation", "SELECT COUNT(*) FROM orders")

    loaded: list = []
    monkeypatch.setattr(openchatbi.config, "load", lambda p: loaded.append(p))
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge([]))

    out_path = tmp_path / "report.json"
    run_judge.run(cases_dir=str(cases_dir), out_path=str(out_path), generated_path=None)
    assert loaded == []


def test_integration_real_example_cases_mocked_evaluator(tmp_path, monkeypatch):
    """Smoke-mode run against the real evals/judge/example_cases with a mocked evaluator."""
    import pathlib

    real_cases = pathlib.Path(__file__).parents[2] / "evals" / "judge" / "example_cases"
    if not real_cases.exists():
        import pytest

        pytest.skip("real cases directory not found")

    calls: list = []
    monkeypatch.setattr(run_judge, "_build_judge", lambda: _make_stub_judge(calls))
    out_path = tmp_path / "report.json"
    rc = run_judge.run(cases_dir=str(real_cases), out_path=str(out_path), generated_path=None)
    assert rc == 0

    report = json.loads(out_path.read_text())
    assert report["mode"] == "smoke"
    # All real cases are evaluated (none skipped in smoke mode).
    assert report["overall"]["skipped"] == 0
    assert report["overall"]["total"] == report["overall"]["evaluated"]
    # All stubs pass.
    assert report["overall"]["pass_rate"] == 1.0
