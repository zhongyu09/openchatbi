"""Unit tests for the RunLedger scripted-trajectory agent driver."""

import glob
import importlib
import os

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _load_agent():
    # Import once and reuse so trajectory tables stay stable across tests.
    return importlib.import_module("evals.runledger.agent.agent")


def test_first_turn_keys_on_prompt_aggregation():
    agent = _load_agent()
    messages = [HumanMessage(content="How many orders were placed in 2024?")]
    out = agent._scripted_llm_call(None, messages)
    assert isinstance(out, AIMessage)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "text2sql"


def test_first_turn_keys_on_prompt_knowledge():
    agent = _load_agent()
    messages = [HumanMessage(content="OpenChatBI")]
    out = agent._scripted_llm_call(None, messages)
    assert out.tool_calls[0]["name"] == "search_knowledge"


def test_trajectory_advances_per_tool_message():
    agent = _load_agent()
    # report case: search_knowledge -> text2sql -> save_report -> final text
    prompt = "Generate a sales report for Q1 2024"
    msgs = [HumanMessage(content=prompt)]
    first = agent._scripted_llm_call(None, msgs)
    assert first.tool_calls[0]["name"] == "search_knowledge"
    msgs += [first, ToolMessage(content="ctx", tool_call_id="c1")]
    second = agent._scripted_llm_call(None, msgs)
    assert second.tool_calls[0]["name"] == "text2sql"


def test_unknown_prompt_falls_back_to_search_knowledge():
    agent = _load_agent()
    out = agent._scripted_llm_call(None, [HumanMessage(content="totally novel question")])
    assert out.tool_calls[0]["name"] == "search_knowledge"


def test_final_turn_emits_no_tool_calls():
    agent = _load_agent()
    prompt = "How many orders were placed in 2024?"
    # aggregation trajectory is single tool then summary
    msgs = [
        HumanMessage(content=prompt),
        AIMessage(content="", tool_calls=[{"name": "text2sql", "args": {}, "id": "c1"}]),
        ToolMessage(content="result", tool_call_id="c1"),
    ]
    out = agent._scripted_llm_call(None, msgs)
    assert out.tool_calls == []


def test_trajectories_prompt_keys_are_unique():
    """Convention #11: assert _TRAJECTORIES prompt keys are unique (no collisions)."""
    agent = _load_agent()
    keys = list(agent._TRAJECTORIES.keys())
    assert len(keys) == len(set(keys)), (
        f"_TRAJECTORIES has duplicate prompt keys: " f"{[k for k in keys if keys.count(k) > 1]}"
    )
    # Also assert the dict length equals the number of distinct prompts
    assert len(agent._TRAJECTORIES) == len(set(keys))


_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "runledger", "cases")
_JUDGE_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "judge", "cases")


def test_corpus_has_at_least_20_cases():
    files = glob.glob(os.path.join(_CASES_DIR, "*.yaml"))
    assert len(files) >= 20, f"Expected >=20 cases, found {len(files)}"


def test_runledger_cases_use_strict_schema():
    for path in glob.glob(os.path.join(_CASES_DIR, "*.yaml")):
        with open(path) as fh:
            case = yaml.safe_load(fh)
        assert set(case) <= {"id", "description", "input", "cassette", "assertions", "budgets"}, path
        assert "category" not in case, path
        assert "gold" not in case, path


def test_judge_cases_have_category_and_gold():
    paths = glob.glob(os.path.join(_JUDGE_CASES_DIR, "*.yaml"))
    assert paths, "No Judge case YAML files found"
    for path in paths:
        with open(path) as fh:
            case = yaml.safe_load(fh)
        assert "category" in case, path
        assert "gold" in case, path
        assert "expected_tool_trajectory" in case["gold"], path


def test_gold_trajectory_matches_driver_table():
    agent = _load_agent()
    for path in glob.glob(os.path.join(_JUDGE_CASES_DIR, "*.yaml")):
        with open(path) as fh:
            case = yaml.safe_load(fh)
        prompt = case["input"]["prompt"]
        driver_traj = [t for t in agent._TRAJECTORIES[prompt] if t is not None]
        assert driver_traj == case["gold"]["expected_tool_trajectory"], path


def test_cassette_args_byte_match_driver():
    """Regression: every cassette tool+args must byte-match what the driver emits.

    This test iterates all cases/*.yaml files that declare a cassette, replays
    each tool turn through _TOOL_ARGS_BUILDERS, and asserts exact equality with
    the corresponding cassette line.  If the save_report builder uses q[:40]
    instead of q[:40].rstrip() the test will fail on c19 and c20.
    """
    import json as _json

    agent = _load_agent()
    _evals_dir = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "runledger")

    case_paths = sorted(glob.glob(os.path.join(_CASES_DIR, "*.yaml")))
    assert case_paths, "No case YAML files found"

    cases_with_cassettes = 0
    for path in case_paths:
        with open(path) as fh:
            case = yaml.safe_load(fh)
        cassette_rel = case.get("cassette")
        if not cassette_rel:
            continue
        cases_with_cassettes += 1

        prompt = case["input"]["prompt"]
        traj = agent._TRAJECTORIES.get(prompt)
        if traj is None:
            continue  # novel prompt — no driver table entry to compare

        tool_calls_in_traj = [t for t in traj if t is not None]

        cassette_path = os.path.join(_evals_dir, cassette_rel)
        cassette_lines = []
        with open(cassette_path) as cf:
            for line in cf:
                line = line.strip()
                if line:
                    cassette_lines.append(_json.loads(line))

        for turn_idx, tool_name in enumerate(tool_calls_in_traj):
            assert turn_idx < len(cassette_lines), (
                f"{os.path.basename(path)} turn {turn_idx}: cassette has only "
                f"{len(cassette_lines)} lines, expected >{turn_idx}"
            )
            cassette_line = cassette_lines[turn_idx]

            # tool name must match
            assert cassette_line["tool"] == tool_name, (
                f"{os.path.basename(path)} turn {turn_idx}: "
                f"cassette tool={cassette_line['tool']!r}, driver={tool_name!r}"
            )

            # args must byte-match (exact dict equality)
            expected_args = agent._TOOL_ARGS_BUILDERS[tool_name](prompt)
            assert cassette_line["args"] == expected_args, (
                f"{os.path.basename(path)} turn {turn_idx} tool={tool_name!r}: "
                f"args mismatch\n"
                f"  driver : {_json.dumps(expected_args)}\n"
                f"  cassette: {_json.dumps(cassette_line['args'])}"
            )

    assert cases_with_cassettes > 0, "No cases with cassettes found — check _CASES_DIR"
