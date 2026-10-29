"""Prove that c21 fails when the real replay graph skips schema discovery.

Run from an environment with this project's eval dependencies installed:
    python evals/runledger/check_regression.py
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals" / "runledger"
CASE_FILE = EVALS / "cases" / "c21_schema_discovery.yaml"


class CheckFailure(RuntimeError):
    """An acceptance condition failed; remains enforced under python -O."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def runtime_environment(suite_dir: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENCHATBI_TEST_MODE"] = "true"
    environment["CONFIG_FILE"] = str(suite_dir / "nonexistent.yaml")
    environment["LANGSMITH_TRACING"] = "false"
    environment["LANGCHAIN_TRACING_V2"] = "false"
    return environment


def resolve_schema_paths(assertions: list[dict]) -> None:
    for assertion in assertions:
        if assertion.get("type") == "json_schema":
            assertion["schema_path"] = str((EVALS / assertion["schema_path"]).resolve())


def prepare_suite(directory: Path, source_suite: dict, source_case: dict, command: list[str]) -> None:
    suite = copy.deepcopy(source_suite)
    case = copy.deepcopy(source_case)
    suite["agent_command"] = command
    suite["mode"] = "replay"
    suite["cases_path"] = "cases"
    # This is a single-case contract probe, not an aggregate pass-rate comparison.
    suite.pop("baseline_path", None)
    suite.pop("regression", None)
    suite.pop("output_dir", None)
    resolve_schema_paths(suite.get("assertions", []))
    resolve_schema_paths(case.get("assertions", []))
    case["cassette"] = str((EVALS / case["cassette"]).resolve())
    (directory / "cases").mkdir(parents=True)
    (directory / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    (directory / "cases" / CASE_FILE.name).write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")


def run_probe(name: str, suite_dir: Path, artifact_dir: Path, schema: dict) -> dict:
    artifact_dir.mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runledger",
            "run",
            str(suite_dir),
            "--mode",
            "replay",
            "--output-dir",
            str(artifact_dir),
        ],
        cwd=ROOT,
        env=runtime_environment(suite_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    (artifact_dir / "console.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    summary_paths = list(artifact_dir.rglob("summary.json"))
    require(len(summary_paths) == 1, f"{name}: expected one summary; see {artifact_dir / 'console.txt'}")
    summary_path = summary_paths[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(len(summary["cases"]) == 1, f"{name}: expected exactly the c21 case")
    case = summary["cases"][0]
    events = [json.loads(line) for line in summary_path.with_name("run.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [event["name"] for event in events if event["type"] == "tool_call"]
    outputs = [event["output"] for event in events if event["type"] == "final_output"]
    require(len(outputs) == 1, f"{name}: the real graph did not emit one final output")
    # Independently check the unchanged output schema, even in the expected-fail run.
    jsonschema.validate(outputs[0], schema)
    failures = [failure for event in events if event["type"] == "assertion_failure" for failure in event["failures"]]
    details = {
        "scenario": name,
        "exit_code": result.returncode,
        "case_status": case["status"],
        "tool_calls": calls,
        "final_schema_passed": True,
        "assertion_failures": failures,
        "failure_reason": case["failure_reason"],
        "artifacts": summary_path.parent.relative_to(ROOT).as_posix(),
    }
    require(case["id"] == "c21_schema_discovery", f"{name}: unexpected case id")
    if name == "unchanged":
        require(result.returncode == 0 and case["status"] == "pass", "Unchanged c21 must pass")
        require(
            calls == ["search_schema"] and case["tool_calls"] == 1, "Unchanged c21 must call search_schema exactly once"
        )
        require(not failures, "Unchanged c21 must have no assertion failures")
    else:
        require(
            result.returncode == 1 and case["status"] == "fail",
            "Skipping schema discovery must exit 1 with a contract failure",
        )
        require(calls == [] and case["tool_calls"] == 0, "Negative probe must make zero tool calls")
        require(
            len(failures) == 1 and failures[0]["type"] == "must_call",
            "Negative probe must fail only the must_call assertion",
        )
        require(
            failures[0].get("details", {}).get("missing") == ["search_schema"],
            "Negative probe must identify search_schema as the missing call",
        )
    details["verified"] = True
    return details


def main() -> int:
    now = datetime.now(UTC)
    evidence_dir = ROOT / "runledger_out" / "schema-regression-check" / now.strftime("%Y%m%d-%H%M%S-%f")
    evidence_dir.mkdir(parents=True)
    evidence = {
        "generated_at": now.isoformat(),
        "passed": False,
        "checks": [],
        "scope": "Local replay graph check with scripted model responses and cassette tool results.",
        "credential_environment_forwarded": False,
    }
    try:
        evidence["runledger_version"] = version("runledger")
        require(
            evidence["runledger_version"] == "0.2.4", "Install this project's pinned runledger==0.2.4 eval dependency"
        )
        source_suite = yaml.safe_load((EVALS / "suite.yaml").read_text(encoding="utf-8"))
        source_case = yaml.safe_load(CASE_FILE.read_text(encoding="utf-8"))
        assertions = source_suite.get("assertions", []) + source_case.get("assertions", [])
        require(
            any(item.get("type") == "must_call" and item.get("tools") == ["search_schema"] for item in assertions),
            "c21 must configure must_call: [search_schema]",
        )
        schema_path = EVALS / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        prompt = source_case["input"]["prompt"]
        with tempfile.TemporaryDirectory(prefix="openchatbi-regression-") as temporary:
            temporary_dir = Path(temporary)
            wrapper = temporary_dir / "skip_schema.py"
            wrapper.write_text(
                "from evals.runledger.agent import agent\n"
                f"agent._TRAJECTORIES[{prompt!r}] = [None]\n"
                "raise SystemExit(agent.main())\n",
                encoding="utf-8",
            )
            commands = {
                "unchanged": [sys.executable, str(EVALS / "agent" / "agent.py")],
                "skip-schema": [sys.executable, str(wrapper)],
            }
            for name, command in commands.items():
                suite_dir = temporary_dir / name
                prepare_suite(suite_dir, source_suite, source_case, command)
                try:
                    check = run_probe(name, suite_dir, evidence_dir / name, schema)
                except Exception as exc:
                    check = {"scenario": name, "verified": False, "error": str(exc)}
                evidence["checks"].append(check)
        evidence["passed"] = all(check["verified"] for check in evidence["checks"])
    except Exception as exc:
        evidence["error"] = str(exc)
    evidence_path = evidence_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    for check in evidence["checks"]:
        if check["verified"]:
            label = (
                "PASS: unchanged graph called search_schema once"
                if check["scenario"] == "unchanged"
                else "PASS: expected must_call failure caught skipped search_schema (exit 1); final schema still valid"
            )
            print(label)
        else:
            print(f"FAIL: {check['scenario']}: {check['error']}")
    if evidence.get("error"):
        print(f"FAIL: {evidence['error']}")
    print(f"Evidence: {evidence_path}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
