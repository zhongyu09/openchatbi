from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any

import yaml

# Load .env before importing openchatbi (the llm_judge import below pulls in
# openchatbi, which instantiates the configured LLM at import time and needs
# ANTHROPIC_API_KEY / OPENAI_API_KEY / CONFIG_FILE in the environment).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from evals.judge.llm_judge import LLMAsJudgeEvaluator


def _build_judge() -> LLMAsJudgeEvaluator:
    # Real LLM judge; the threshold is config-adjustable per S2.
    return LLMAsJudgeEvaluator(threshold=0.7)


def _load_cases(cases_dir: str) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.yaml"))):
        with open(path) as fh:
            case = yaml.safe_load(fh) or {}
        if "gold" in case and case["gold"].get("expected_sql"):
            cases.append(case)
    return cases


def _load_generated_map(generated_path: str) -> dict[str, str]:
    """Load agent-generated SQL from a JSON map or JSONL file.

    Supports two shapes:
    - JSON object: ``{"<case_id_or_prompt>": "SELECT ..."}``
    - JSONL lines:  ``{"id": "...", "prompt": "...", "generated_sql": "SELECT ..."}``

    Returns a dict keyed by both ``id`` and ``prompt`` (when available) so
    callers can look up by either.
    """
    with open(generated_path) as fh:
        raw = fh.read().strip()

    result: dict[str, str] = {}

    # Try JSON object first — but only when it looks like a simple id→sql map,
    # i.e. NOT a JSONL record (which would have a "generated_sql" key itself).
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "generated_sql" not in obj:
                # Simple map: {"<case_id_or_prompt>": "SELECT ...", ...}
                result.update(obj)
                return result
        except json.JSONDecodeError:
            pass  # fall through to JSONL

    # JSONL: one JSON object per line.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[run_judge] Warning: Skipping malformed JSON line in generated file: {exc}", file=sys.stderr)
            continue
            
        sql = record.get("generated_sql", "")
        if record.get("id"):
            result[record["id"]] = sql
        if record.get("prompt"):
            result[record["prompt"]] = sql

    return result


def _build_report(mode: str, results: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    # Aggregate per category — skipped cases are counted but excluded from
    # pass_rate and mean_score.
    by_category: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_category.setdefault(
            r["category"],
            {"scores": [], "passed": 0, "skipped": 0, "total": 0, "evaluated": 0},
        )
        bucket["total"] += 1
        if r["skipped"]:
            bucket["skipped"] += 1
        else:
            bucket["scores"].append(r["score"])
            bucket["passed"] += 1 if r["passed"] else 0
            bucket["evaluated"] += 1

    for _cat, bucket in by_category.items():
        scores = bucket.pop("scores")
        evaluated = bucket["evaluated"]
        bucket["mean_score"] = statistics.mean(scores) if scores else 0.0
        bucket["pass_rate"] = bucket["passed"] / evaluated if evaluated else 0.0

    evaluated_results = [r for r in results if not r["skipped"]]
    overall_passed = sum(1 for r in evaluated_results if r["passed"])
    overall_evaluated = len(evaluated_results)
    overall_skipped = len(results) - overall_evaluated

    return {
        "mode": mode,
        "progress": {
            "processed": len(results),
            "total": total_cases,
            "complete": len(results) == total_cases,
        },
        "overall": {
            "total": len(results),
            "evaluated": overall_evaluated,
            "skipped": overall_skipped,
            "passed": overall_passed,
            "pass_rate": (overall_passed / overall_evaluated) if overall_evaluated else 0.0,
            "mean_score": (statistics.mean([r["score"] for r in evaluated_results]) if evaluated_results else 0.0),
        },
        "by_category": by_category,
        "cases": results,
    }


def _write_report(mode: str, results: list[dict[str, Any]], out_path: str, total_cases: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(_build_report(mode, results, total_cases), fh, indent=2)
    os.replace(tmp_path, out_path)


def _short_prompt(prompt: str, max_chars: int = 88) -> str:
    one_line = " ".join(prompt.split())
    if len(one_line) <= max_chars:
        return one_line
    return f"{one_line[: max_chars - 3]}..."


def print_judge_plan(
    cases: list[dict[str, Any]],
    out_path: str,
    mode: str,
    generated_path: str | None,
    stream: Any = sys.stderr,
) -> None:
    """Print a startup summary before judge evaluation begins."""
    print("=== Judge Evaluation Plan ===", file=stream)
    print(f"Mode: {mode}", file=stream)
    if generated_path:
        print(f"Generated SQL: {generated_path}", file=stream)
    else:
        print("Generated SQL: not supplied; using gold SQL for smoke checks.", file=stream)
    print(f"Output: {out_path}", file=stream)
    print(f"Total cases: {len(cases)}", file=stream)
    print("=============================", file=stream)


def print_case_start(case: dict[str, Any], idx: int, total: int, stream: Any = sys.stderr) -> None:
    """Print a visible progress message before judging one case."""
    case_id = str(case.get("id", ""))
    prompt = _short_prompt(str((case.get("input") or {}).get("prompt", "")))
    bar = "=" * 64
    print("", file=stream)
    print(bar, file=stream)
    print(f"JUDGING CASE {idx}/{total}: {case_id}", file=stream)
    if prompt:
        print(f"Prompt: {prompt}", file=stream)
    print(bar, file=stream)


def print_case_progress(
    result: dict[str, Any],
    processed: int,
    total: int,
    out_path: str,
    stream: Any = sys.stderr,
) -> None:
    """Print progress after the incremental report has been persisted."""
    if result["skipped"]:
        status = f"skipped: {result.get('skip_reason', 'unknown')}"
    else:
        verdict = "passed" if result["passed"] else "failed"
        status = f"{verdict}, score={result['score']:.3f}"
    print(f"judged {processed}/{total} cases ({status}) -> {out_path}", file=stream)


def run(
    cases_dir: str,
    out_path: str,
    generated_path: str | None = None,
    config_path: str | None = None,
) -> int:
    # Point openchatbi at the requested config BEFORE building the judge, so the
    # LLM judge uses the same default_llm as the agent (run_judge has no implicit
    # config beyond $CONFIG_FILE otherwise).
    if config_path:
        from openchatbi import config as _config

        _config.load(config_path)

    judge = _build_judge()
    cases = _load_cases(cases_dir)

    # Determine mode and emit warning when doing smoke check.
    if generated_path is None:
        mode = "smoke"
        print(
            "[run_judge] no --generated supplied: running gold-vs-gold SMOKE check "
            "(scores are not meaningful). Pass --generated <map> to evaluate real agent SQL.",
            file=sys.stderr,
        )
        generated_map: dict[str, str] = {}
    else:
        mode = "generated"
        generated_map = _load_generated_map(generated_path)

    print_judge_plan(cases, out_path, mode, generated_path, stream=sys.stderr)

    results: list[dict[str, Any]] = []
    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        gold = case["gold"]
        prompt: str = case["input"]["prompt"]
        case_id: str = case.get("id", "")
        result: dict[str, Any]

        print_case_start(case, idx, total_cases, stream=sys.stderr)

        if mode == "smoke":
            # Gold-vs-gold smoke check — intentional wiring/baseline test.
            # Both generated_sql and expected_sql are set to the gold SQL so
            # that the judge always receives a near-perfect pair, confirming
            # the full evaluation pipeline is wired correctly.
            verdict = judge.judge(
                question=prompt,
                generated_sql=gold["expected_sql"],
                expected_sql=gold["expected_sql"],
            )
            result = {
                "id": case_id,
                "category": case.get("category", "uncategorized"),
                "score": verdict.score,
                "passed": verdict.passed,
                "skipped": False,
                "reasoning": verdict.reasoning,
            }
        else:
            # Look up by case id first, fall back to prompt.
            generated_sql = generated_map.get(case_id) or generated_map.get(prompt)

            if generated_sql is None:
                result = {
                    "id": case_id,
                    "category": case.get("category", "uncategorized"),
                    "score": None,
                    "passed": False,
                    "skipped": True,
                    "skip_reason": "no_generated_sql",
                    "reasoning": "skipped: no generated SQL found for this case",
                }
                results.append(result)
                _write_report(mode, results, out_path, total_cases=total_cases)
                print_case_progress(result, len(results), total_cases, out_path, stream=sys.stderr)
                continue

            verdict = judge.judge(
                question=prompt,
                generated_sql=generated_sql,
                expected_sql=gold["expected_sql"],
            )
            result = {
                "id": case_id,
                "category": case.get("category", "uncategorized"),
                "score": verdict.score,
                "passed": verdict.passed,
                "skipped": False,
                "reasoning": verdict.reasoning,
            }

        results.append(result)
        _write_report(mode, results, out_path, total_cases=total_cases)
        print_case_progress(result, len(results), total_cases, out_path, stream=sys.stderr)

    _write_report(mode, results, out_path, total_cases=total_cases)
    print(f"Judge evaluation complete: {len(results)}/{total_cases} cases -> {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.judge.run_judge")
    parser.add_argument("--cases", default="evals/judge/example_cases")
    parser.add_argument("--out", default="judge_out/report.json")
    parser.add_argument(
        "--generated",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON map or JSONL file supplying agent-generated SQL per case. "
            "Supported shapes: "
            '{"<case_id_or_prompt>": "SELECT ..."} '
            'or JSONL lines {"id": "...", "prompt": "...", "generated_sql": "SELECT ..."}. '
            "When omitted, runs a gold-vs-gold SMOKE check instead."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to the openchatbi config yaml the LLM judge should use "
            "(same one passed to collect_generated). When omitted, falls back to "
            "$CONFIG_FILE / the default config."
        ),
    )
    args = parser.parse_args(argv)
    return run(
        cases_dir=args.cases,
        out_path=args.out,
        generated_path=args.generated,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
