"""Unit tests for the RunLedger scripted-trajectory agent driver."""

import glob
import importlib
import os

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _load_agent():
    # agent.py mutates builtins.print on import; import once and reuse.
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
        f"_TRAJECTORIES has duplicate prompt keys: "
        f"{[k for k in keys if keys.count(k) > 1]}"
    )
    # Also assert the dict length equals the number of distinct prompts
    assert len(agent._TRAJECTORIES) == len(set(keys))


_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "runledger", "cases")


def test_corpus_has_at_least_20_cases():
    files = glob.glob(os.path.join(_CASES_DIR, "*.yaml"))
    assert len(files) >= 20, f"Expected >=20 cases, found {len(files)}"


def test_every_case_has_category_and_gold():
    for path in glob.glob(os.path.join(_CASES_DIR, "*.yaml")):
        if os.path.basename(path) == "t1.yaml":
            continue  # seed case predates the gold schema
        with open(path) as fh:
            case = yaml.safe_load(fh)
        assert "category" in case, path
        assert "gold" in case, path
        assert "expected_tool_trajectory" in case["gold"], path


def test_gold_trajectory_matches_driver_table():
    agent = _load_agent()
    for path in glob.glob(os.path.join(_CASES_DIR, "*.yaml")):
        if os.path.basename(path) == "t1.yaml":
            continue
        with open(path) as fh:
            case = yaml.safe_load(fh)
        prompt = case["input"]["prompt"]
        driver_traj = [t for t in agent._TRAJECTORIES[prompt] if t is not None]
        assert driver_traj == case["gold"]["expected_tool_trajectory"], path
