from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any

import yaml

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


def run(cases_dir: str, out_path: str) -> int:
    judge = _build_judge()
    cases = _load_cases(cases_dir)
    results: list[dict[str, Any]] = []
    for case in cases:
        gold = case["gold"]
        verdict = judge.judge(
            question=case["input"]["prompt"],
            generated_sql=gold["expected_sql"],
            expected_sql=gold["expected_sql"],
        )
        results.append(
            {
                "id": case["id"],
                "category": case.get("category", "uncategorized"),
                "score": verdict.score,
                "passed": verdict.passed,
                "reasoning": verdict.reasoning,
            }
        )

    by_category: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_category.setdefault(r["category"], {"scores": [], "passed": 0, "total": 0})
        bucket["scores"].append(r["score"])
        bucket["passed"] += 1 if r["passed"] else 0
        bucket["total"] += 1
    for cat, bucket in by_category.items():
        scores = bucket.pop("scores")
        bucket["mean_score"] = statistics.mean(scores) if scores else 0.0
        bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0

    overall_passed = sum(1 for r in results if r["passed"])
    report = {
        "overall": {
            "total": len(results),
            "passed": overall_passed,
            "pass_rate": (overall_passed / len(results)) if results else 0.0,
            "mean_score": statistics.mean([r["score"] for r in results]) if results else 0.0,
        },
        "by_category": by_category,
        "cases": results,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.judge.run_judge")
    parser.add_argument("--cases", default="evals/runledger/cases")
    parser.add_argument("--out", default="judge_out/report.json")
    args = parser.parse_args(argv)
    return run(cases_dir=args.cases, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
