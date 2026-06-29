"""Unit tests for evals/judge/collect_generated.py.

Hermetic: no real LLM, no network, no graph build. ``runner`` is injected as a
fake so we exercise the pure functions and IO only.
"""

import io
import json

from evals.judge import collect_generated as cg
from evals.judge import run_judge
from openchatbi.observability.context import current_request_id, current_user_id, get_run_context

# ---------------------------------------------------------------------------
# extract_sql_from_state
# ---------------------------------------------------------------------------


def test_extract_sql_reads_key():
    state = {"sql": "SELECT COUNT(*) FROM orders"}
    assert cg.extract_sql_from_state(state) == "SELECT COUNT(*) FROM orders"


def test_extract_sql_interrupt_returns_empty():
    state = {"__interrupt__": [{"value": {"text": "approve?"}}], "sql": "SELECT 1"}
    assert cg.extract_sql_from_state(state) == ""


def test_extract_sql_missing_key_returns_empty():
    assert cg.extract_sql_from_state({"messages": []}) == ""


def test_extract_sql_strips_whitespace():
    assert cg.extract_sql_from_state({"sql": "  SELECT 1  \n"}) == "SELECT 1"


def test_extract_sql_custom_key():
    assert cg.extract_sql_from_state({"final_sql": "SELECT 2"}, sql_key="final_sql") == "SELECT 2"


def test_extract_sql_none_value():
    assert cg.extract_sql_from_state({"sql": None}) == ""


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def test_collect_canned_states():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "c02", "input": {"prompt": "q2"}},
    ]
    canned = {
        "c01": {"sql": "SELECT 1"},
        "c02": {"sql": "SELECT 2"},
    }

    def runner(case):
        return canned[case["id"]]

    records = cg.collect(cases, runner)
    assert records == [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"},
        {"id": "c02", "prompt": "q2", "generated_sql": "SELECT 2"},
    ]


def test_collect_per_case_isolation_on_exception():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "boom", "input": {"prompt": "q2"}},
        {"id": "c03", "input": {"prompt": "q3"}},
    ]

    def runner(case):
        if case["id"] == "boom":
            raise RuntimeError("agent blew up")
        return {"sql": f"SELECT '{case['id']}'"}

    records = cg.collect(cases, runner)
    assert records[0]["generated_sql"] == "SELECT 'c01'"
    # The failing case is recorded with empty sql, others unaffected.
    assert records[1] == {"id": "boom", "prompt": "q2", "generated_sql": ""}
    assert records[2]["generated_sql"] == "SELECT 'c03'"


def test_collect_interrupt_state_yields_empty_sql():
    cases = [{"id": "c01", "input": {"prompt": "q1"}}]

    def runner(case):
        return {"__interrupt__": [{"value": {}}]}

    records = cg.collect(cases, runner)
    assert records[0]["generated_sql"] == ""


def test_collect_calls_on_update_after_each_record():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "c02", "input": {"prompt": "q2"}},
    ]
    snapshots = []

    def runner(case):
        return {"sql": f"SELECT '{case['id']}'"}

    def on_update(records):
        snapshots.append(list(records))

    records = cg.collect(cases, runner, on_update=on_update)

    assert records == [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 'c01'"},
        {"id": "c02", "prompt": "q2", "generated_sql": "SELECT 'c02'"},
    ]
    assert [[record["id"] for record in snapshot] for snapshot in snapshots] == [["c01"], ["c01", "c02"]]


def test_collect_calls_on_case_start_before_each_record():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "c02", "input": {"prompt": "q2"}},
    ]
    events = []

    def runner(case):
        events.append(("run", case["id"]))
        return {"sql": f"SELECT '{case['id']}'"}

    def on_case_start(case, idx, total):
        events.append(("start", case["id"], idx, total))

    cg.collect(cases, runner, on_case_start=on_case_start)

    assert events == [
        ("start", "c01", 1, 2),
        ("run", "c01"),
        ("start", "c02", 2, 2),
        ("run", "c02"),
    ]


# ---------------------------------------------------------------------------
# _state_from_graph — reads terminal SQL from the checkpointer, NOT invoke()
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Mimics the compiled agent graph streaming: text2sql runs as a subgraph,
    so the committed SQL only appears in the `generate_sql`/`regenerate_sql`
    stream updates — never in a terminal state."""

    def __init__(self, updates):
        self._updates = updates  # list of (namespace, update_dict)
        self.stream_calls = []
        self.run_contexts = []

    def stream(self, payload, config, stream_mode=None, subgraphs=None):
        self.stream_calls.append((payload, config, stream_mode, subgraphs))
        self.run_contexts.append(get_run_context())
        return iter(self._updates)


def test_state_from_graph_reads_sql_from_stream_updates():
    # generate_sql update carries the SQL; later non-sql nodes don't clobber it.
    graph = _FakeGraph(
        [
            (("text2sql:1",), {"generate_sql": {"sql": "SELECT 1"}}),
            (("text2sql:1",), {"execute_sql": {"sql_execution_result": "SQL_SUCCESS"}}),
        ]
    )
    state = cg._state_from_graph(graph, {"id": "c1", "input": {"prompt": "q"}})
    assert cg.extract_sql_from_state(state) == "SELECT 1"
    # streamed with the right thread_id + subgraphs=True.
    payload, config, stream_mode, subgraphs = graph.stream_calls[0]
    assert config["configurable"]["thread_id"] == "eval-c1"
    assert config["configurable"]["user_id"] == "eval-c1"
    assert config["metadata"]["user_id"] == "eval-c1"
    assert config["metadata"]["request_id"] == "eval-c1"
    assert payload["messages"][0]["content"] == "q"
    assert subgraphs is True
    assert graph.run_contexts == [("eval-c1", "eval-c1")]


def test_state_from_graph_restores_previous_run_context():
    user_token = current_user_id.set("outer-user")
    request_token = current_request_id.set("outer-request")
    graph = _FakeGraph([(("text2sql:1",), {"generate_sql": {"sql": "SELECT 1"}})])

    try:
        cg._state_from_graph(graph, {"id": "c1", "input": {"prompt": "q"}})

        assert get_run_context() == ("outer-user", "outer-request")
    finally:
        current_request_id.reset(request_token)
        current_user_id.reset(user_token)


def test_state_from_graph_regenerate_wins_over_first_sql():
    # A retry path: regenerate_sql emits later, so its SQL is the final one.
    graph = _FakeGraph(
        [
            (("text2sql:1",), {"generate_sql": {"sql": "SELECT bad"}}),
            (("text2sql:1",), {"regenerate_sql": {"sql": "SELECT good"}}),
        ]
    )
    state = cg._state_from_graph(graph, {"id": "c1", "input": {"prompt": "q"}})
    assert cg.extract_sql_from_state(state) == "SELECT good"


def test_state_from_graph_marks_interrupt_when_paused():
    # Paused at ask_human: stream yields an __interrupt__ update, no SQL committed.
    graph = _FakeGraph([((), {"__interrupt__": [{"value": {"text": "approve?"}}]})])
    state = cg._state_from_graph(graph, {"id": "c2", "input": {"prompt": "q"}})
    assert state.get("__interrupt__") is True
    assert cg.extract_sql_from_state(state) == ""


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------


def test_write_output_json_is_id_sql_map(tmp_path):
    records = [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"},
        {"id": "c02", "prompt": "q2", "generated_sql": "SELECT 2"},
    ]
    out = tmp_path / "nested" / "generated.json"
    cg.write_output(records, str(out), fmt="json")
    obj = json.loads(out.read_text())
    assert obj == {"c01": "SELECT 1", "c02": "SELECT 2"}


def test_write_output_jsonl_has_full_records(tmp_path):
    records = [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"},
        {"id": "c02", "prompt": "q2", "generated_sql": "SELECT 2"},
    ]
    out = tmp_path / "generated.jsonl"
    cg.write_output(records, str(out), fmt="jsonl")
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    for rec, line in zip(records, lines, strict=False):
        assert line["id"] == rec["id"]
        assert line["prompt"] == rec["prompt"]
        assert line["generated_sql"] == rec["generated_sql"]


def test_write_progress_records_completion_state(tmp_path):
    out = tmp_path / "generated.json"
    cg.write_progress(str(out), processed=1, total=2)
    progress = json.loads((tmp_path / "generated.json.progress.json").read_text())
    assert progress == {
        "processed": 1,
        "total": 2,
        "complete": False,
        "out": str(out),
    }


def test_load_existing_output_json_reads_id_sql_map(tmp_path):
    out = tmp_path / "generated.json"
    out.write_text(json.dumps({"c01": "SELECT 1", "c02": ""}))

    assert cg.load_existing_output(str(out), "json") == {"c01": "SELECT 1", "c02": ""}


def test_load_existing_output_jsonl_reads_records_by_id(tmp_path):
    out = tmp_path / "generated.jsonl"
    out.write_text(
        "\n".join(
            [
                json.dumps({"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"}),
                json.dumps({"id": "c02", "prompt": "q2", "generated_sql": ""}),
            ]
        )
    )

    assert cg.load_existing_output(str(out), "jsonl") == {"c01": "SELECT 1", "c02": ""}


def test_records_from_existing_preserves_case_order():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "c02", "input": {"prompt": "q2"}},
        {"id": "c03", "input": {"prompt": "q3"}},
    ]

    records = cg.records_from_existing(cases, {"c02": "", "c01": "SELECT 1"})

    assert records == [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"},
        {"id": "c02", "prompt": "q2", "generated_sql": ""},
    ]


def test_merge_records_by_case_order_keeps_existing_and_new_records():
    cases = [
        {"id": "c01", "input": {"prompt": "q1"}},
        {"id": "c02", "input": {"prompt": "q2"}},
        {"id": "c03", "input": {"prompt": "q3"}},
    ]
    existing = {"c01": "SELECT 1"}
    new_records = [{"id": "c03", "prompt": "q3", "generated_sql": "SELECT 3"}]

    assert cg.merge_records_by_case_order(cases, existing, new_records) == [
        {"id": "c01", "prompt": "q1", "generated_sql": "SELECT 1"},
        {"id": "c03", "prompt": "q3", "generated_sql": "SELECT 3"},
    ]


def test_print_collection_plan_lists_completed_empty_and_pending(tmp_path):
    cases = [
        {"id": "c01", "input": {"prompt": "Already done?"}},
        {"id": "c02", "input": {"prompt": "Executed but no sql?"}},
        {"id": "c03", "input": {"prompt": "Not yet?"}},
    ]
    out = tmp_path / "generated.json"
    out.write_text(json.dumps({"c01": "SELECT 1", "c02": ""}))
    stream = io.StringIO()

    cg.print_collection_plan(cases, str(out), "json", stream=stream)

    log = stream.getvalue()
    assert "Total: 3 | Done: 1 | Done with empty SQL: 1 | Pending: 1" in log
    assert "[done] c01" in log
    assert "[done, empty SQL] c02" in log
    assert "[pending] c03" in log
    assert "Already done?" in log


def test_print_case_start_is_visible_and_user_friendly():
    stream = io.StringIO()
    case = {"id": "c01", "input": {"prompt": "List ten customers with their id and name."}}

    cg.print_case_start(case, 1, 3, stream=stream)

    log = stream.getvalue()
    assert "================================================" in log
    assert "RUNNING CASE 1/3: c01" in log
    assert "Prompt: List ten customers with their id and name." in log


# ---------------------------------------------------------------------------
# load_cases
# ---------------------------------------------------------------------------


def _write_case(tmp_dir, name, prompt, with_gold=True):
    p = tmp_dir / f"{name}.yaml"
    text = f"id: {name}\n" "category: test\n" "input:\n" f"  prompt: '{prompt}'\n"
    if with_gold:
        text += 'gold:\n  expected_sql: "SELECT 1"\n'
    p.write_text(text)
    return p


def test_load_cases_reads_id_and_prompt(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "b_second", "second question")
    _write_case(cases_dir, "a_first", "first question", with_gold=False)

    cases = cg.load_cases(str(cases_dir))
    assert len(cases) == 2
    # Sorted by filename: a_first before b_second.
    assert cases[0]["id"] == "a_first"
    assert cases[0]["input"]["prompt"] == "first question"
    assert cases[1]["id"] == "b_second"
    assert cases[1]["input"]["prompt"] == "second question"
    # gold is NOT required — caseless prompt file still loads.
    assert "gold" not in cases[0]


def test_main_writes_generated_and_progress_incrementally(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "How many orders?")
    _write_case(cases_dir, "c02", "Total revenue?")

    def fake_runner(case):
        return {"sql": f"SELECT /* {case['id']} */ 1"}

    monkeypatch.setattr(cg, "build_agent_runner", lambda config_path: fake_runner)
    out = tmp_path / "generated.json"

    rc = cg.main(
        [
            "--cases",
            str(cases_dir),
            "--config",
            "fake_config.yaml",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text()) == {
        "c01": "SELECT /* c01 */ 1",
        "c02": "SELECT /* c02 */ 1",
    }
    assert json.loads((tmp_path / "generated.json.progress.json").read_text())["complete"] is True


def test_main_resumes_from_existing_output(tmp_path, monkeypatch, capsys):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_case(cases_dir, "c01", "First?")
    _write_case(cases_dir, "c02", "Second?")
    _write_case(cases_dir, "c03", "Third?")

    called = []

    def fake_runner(case):
        called.append(case["id"])
        return {"sql": f"SELECT /* {case['id']} */ 1"}

    monkeypatch.setattr(cg, "build_agent_runner", lambda config_path: fake_runner)
    out = tmp_path / "generated.json"
    out.write_text(json.dumps({"c01": "SELECT 1", "c02": ""}))

    rc = cg.main(
        [
            "--cases",
            str(cases_dir),
            "--config",
            "fake_config.yaml",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert called == ["c03"]
    assert json.loads(out.read_text()) == {
        "c01": "SELECT 1",
        "c02": "",
        "c03": "SELECT /* c03 */ 1",
    }
    progress = json.loads((tmp_path / "generated.json.progress.json").read_text())
    assert progress["processed"] == 3
    assert progress["complete"] is True
    assert "RUNNING CASE 3/3: c03" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ROUND-TRIP: output is consumable by run_judge._load_generated_map
# ---------------------------------------------------------------------------


def test_roundtrip_json_consumable_by_judge(tmp_path):
    cases = [
        {"id": "c01", "input": {"prompt": "How many orders?"}},
        {"id": "c02", "input": {"prompt": "Total revenue?"}},
    ]

    def runner(case):
        return {"sql": f"SELECT /* {case['id']} */ 1"}

    records = cg.collect(cases, runner)
    out = tmp_path / "generated.json"
    cg.write_output(records, str(out), fmt="json")

    gen_map = run_judge._load_generated_map(str(out))
    assert gen_map.get("c01") == "SELECT /* c01 */ 1"
    assert gen_map.get("c02") == "SELECT /* c02 */ 1"


def test_roundtrip_jsonl_consumable_by_judge(tmp_path):
    cases = [
        {"id": "c01", "input": {"prompt": "How many orders?"}},
        {"id": "c02", "input": {"prompt": "Total revenue?"}},
    ]

    def runner(case):
        return {"sql": f"SELECT /* {case['id']} */ 1"}

    records = cg.collect(cases, runner)
    out = tmp_path / "generated.jsonl"
    cg.write_output(records, str(out), fmt="jsonl")

    gen_map = run_judge._load_generated_map(str(out))
    # JSONL is keyed by both id and prompt.
    assert gen_map.get("c01") == "SELECT /* c01 */ 1"
    assert gen_map.get("How many orders?") == "SELECT /* c01 */ 1"
    assert gen_map.get("c02") == "SELECT /* c02 */ 1"
    assert gen_map.get("Total revenue?") == "SELECT /* c02 */ 1"
