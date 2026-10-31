import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset
from src.data.profiler import profile_dataset
from src.llm.client import get_llm_client, MockLLMClient, OllamaClient, GeminiClient
from src.llm.planner import AnalysisPlanner
from src.analysis.executor import execute_plan


def run_benchmark(
    benchmark_path: str = "evaluation/benchmark_questions.json",
    provider: str = "mock",
) -> Dict[str, Any]:
    """Runs the benchmark suite against real datasets and scores planner accuracy."""
    with open(benchmark_path, "r") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    # Initialize client and planner
    client = get_llm_client(provider=provider)
    planner = AnalysisPlanner(client=client)

    results = []
    dataset_cache: Dict[str, Any] = {}

    print(f"\n=======================================================")
    print(f"STARTING ANALYSIS PLANNER BENCHMARK ({len(questions)} Questions)")
    print(f"Provider: {client.__class__.__name__}")
    print(f"=======================================================\n")

    for item in questions:
        q_id = item["id"]
        ds_path = item["dataset"]
        sheet = item.get("sheet", 0)
        tier = item.get("tier", "medium")
        question = item["question"]

        # Cache loaded & profiled datasets
        cache_key = f"{ds_path}_{sheet}"
        if cache_key not in dataset_cache:
            raw_df = load_dataset(ds_path, sheet_name=sheet)
            cleaned_df, _ = clean_dataset(raw_df)
            prof = profile_dataset(cleaned_df)
            dataset_cache[cache_key] = (cleaned_df, prof)

        df, profile = dataset_cache[cache_key]

        # 1. Planning Phase
        planning_passed = False
        execution_passed = False
        failure_category = None
        plan = None
        exec_result = None

        try:
            plan = planner.plan(question, profile)
            # Verify planned operation
            if plan.operation.value == item["expected_operation"]:
                planning_passed = True
            else:
                failure_category = "wrong_operation"

            # Check target column if expected
            if planning_passed and item.get("expected_target_column"):
                expected_tgt = str(item["expected_target_column"]).lower()
                planned_tgt = str(plan.target_column or "").lower()
                if expected_tgt != planned_tgt and planned_tgt != "":
                    planning_passed = False
                    failure_category = "wrong_column"

            # Check group by if expected
            if planning_passed and item.get("expected_group_by"):
                exp_grp = [g.lower() for g in item["expected_group_by"]]
                plan_grp = [g.lower() for g in (plan.group_by_columns or [])]
                if set(exp_grp) != set(plan_grp):
                    planning_passed = False
                    failure_category = "wrong_column"

        except Exception as pe:
            failure_category = f"invalid_plan ({str(pe)})"

        # 2. Execution Phase
        if plan is not None:
            try:
                exec_result = execute_plan(plan, df)
                execution_passed = True
            except Exception as ee:
                execution_passed = False
                failure_category = f"execution_failure ({str(ee)})"

        results.append({
            "id": q_id,
            "tier": tier,
            "question": question,
            "planning_passed": planning_passed,
            "execution_passed": execution_passed,
            "failure_category": failure_category,
            "main_value": exec_result.main_value if exec_result else None,
        })

        status_str = "PASS" if (planning_passed and execution_passed) else "FAIL"
        print(f"[{status_str}] {q_id} ({tier.upper()}): '{question}' -> Plan: {plan.operation.value if plan else 'None'}")

    # Aggregated Scorecard
    total_q = len(results)
    plan_pass_cnt = sum(1 for r in results if r["planning_passed"])
    exec_pass_cnt = sum(1 for r in results if r["execution_passed"])
    both_pass_cnt = sum(1 for r in results if r["planning_passed"] and r["execution_passed"])

    planning_accuracy = (plan_pass_cnt / max(1, total_q)) * 100.0
    execution_rate = (exec_pass_cnt / max(1, total_q)) * 100.0
    overall_accuracy = (both_pass_cnt / max(1, total_q)) * 100.0

    print(f"\n=======================================================")
    print(f"BENCHMARK SCORECARD SUMMARY")
    print(f"=======================================================")
    print(f"Total Benchmark Questions: {total_q}")
    print(f"Planning Accuracy:        {plan_pass_cnt}/{total_q} ({planning_accuracy:.1f}%)")
    print(f"Execution Success Rate:   {exec_pass_cnt}/{total_q} ({execution_rate:.1f}%)")
    print(f"End-to-End Correctness:   {both_pass_cnt}/{total_q} ({overall_accuracy:.1f}%)")
    print(f"Target Benchmark Gate:    >= 85.0%")

    return {
        "total_questions": total_q,
        "planning_accuracy": planning_accuracy,
        "execution_rate": execution_rate,
        "overall_accuracy": overall_accuracy,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock", help="LLM provider (mock, ollama, gemini)")
    args = parser.parse_args()
    summary = run_benchmark(provider=args.provider)
    if summary["overall_accuracy"] < 85.0:
        print("\nBenchmark gate failed (<85%).")
        sys.exit(1)
    else:
        print("\nBenchmark gate PASSED (>=85%).")
        sys.exit(0)
