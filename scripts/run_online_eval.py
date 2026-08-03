"""CLI: run online eval for a candidate model against a baseline (Part 3).

Runs the deterministic workflow eval fixtures (``eval_runner.run_eval``) for the
candidate, compares against the latest baseline ``EvalRun`` stored locally, and
persists the result. Prints the pass rate and per-metric deltas.

Usage:
    python scripts/run_online_eval.py \
        --baseline-db eval_runs.db \
        --model-version aca-20260803-a1b2 \
        --baseline-version aca-20260801-c0de \
        --target-categories retrieval_miss,hallucination
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "agent-runtime"))

from allcallall_agent_runtime.eval_runner import WorkflowEvalReport, run_eval  # noqa: E402
from allcallall_agent_runtime.models import BadcaseCategory  # noqa: E402
from allcallall_agent_runtime.online_eval import EvalRunStore, build_eval_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run online eval for a candidate vs baseline.")
    parser.add_argument("--baseline-db", default="eval_runs.db", help="EvalRunStore SQLite path")
    parser.add_argument("--model-version", default="candidate", help="Candidate model version")
    parser.add_argument("--baseline-version", default="baseline", help="Baseline model version")
    parser.add_argument("--dataset-ref", default="evals/cases.json", help="Eval dataset reference")
    parser.add_argument(
        "--target-categories",
        default="",
        help="Comma-separated BadcaseCategory values to verify improvement on",
    )
    args = parser.parse_args()

    candidate_report = run_eval()
    baseline: WorkflowEvalReport | None = None
    if Path(args.baseline_db).exists():
        store = EvalRunStore(args.baseline_db)
        latest = store.latest_for_version(args.baseline_version)
        if latest is not None:
            baseline = WorkflowEvalReport(summary=latest.metrics, cases=[])

    targets = [BadcaseCategory(v) for v in args.target_categories.split(",") if v]
    run = build_eval_run(
        candidate_report,
        baseline,
        model_version=args.model_version,
        baseline_version=args.baseline_version,
        dataset_ref=args.dataset_ref,
        dataset_kind="golden",
        target_categories=targets,
    )
    EvalRunStore(args.baseline_db).save(run)
    print(
        f"model={run.model_version} baseline={run.baseline_version} "
        f"improved={run.improved} "
        f"passed={run.metrics.passed_cases}/{run.metrics.total_cases}"
    )
    for key, value in run.delta_vs_baseline.items():
        print(f"  delta {key}: {value:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
