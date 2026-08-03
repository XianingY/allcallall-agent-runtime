"""Tests for Part 3 online evaluation: comparison logic and EvalRunStore."""

from __future__ import annotations

from allcallall_agent_runtime.engineering_harness import RagEvalCase
from allcallall_agent_runtime.models import (
    BadcaseCategory,
    WorkflowEvalReport,
    WorkflowEvalSummary,
)
from allcallall_agent_runtime.online_eval import (
    EvalRunStore,
    build_eval_run,
    compare_eval_runs,
    rag_metrics_from_cases,
)


def _summary(**kw: float) -> WorkflowEvalSummary:
    return WorkflowEvalSummary(
        total_cases=10,
        passed_cases=int(kw.get("passed_cases", 10)),
        task_success_rate=kw.get("task_success_rate", 1.0),
        citation_grounding_rate=kw.get("citation_grounding_rate", 1.0),
        approval_safety_rate=kw.get("approval_safety_rate", 1.0),
        route_accuracy=kw.get("route_accuracy", 1.0),
        grounding_check_rate=kw.get("grounding_check_rate", 1.0),
        unsupported_claim_guard_rate=kw.get("unsupported_claim_guard_rate", 1.0),
    )


def test_compare_improved_when_target_metric_up() -> None:
    baseline = _summary(grounding_check_rate=0.5, passed_cases=5)
    candidate = _summary(grounding_check_rate=0.9, passed_cases=9)
    delta, improved = compare_eval_runs(baseline, candidate, [BadcaseCategory.RETRIEVAL_MISS])
    assert improved is True
    assert delta["grounding_check_rate"] == 0.4


def test_compare_not_improved_when_target_metric_down() -> None:
    baseline = _summary(grounding_check_rate=0.9, passed_cases=9)
    candidate = _summary(grounding_check_rate=0.5, passed_cases=5)
    _, improved = compare_eval_runs(baseline, candidate, [BadcaseCategory.RETRIEVAL_MISS])
    assert improved is False


def test_compare_not_improved_on_overall_regression() -> None:
    baseline = _summary(grounding_check_rate=0.5, passed_cases=10)
    candidate = _summary(grounding_check_rate=0.9, passed_cases=3)
    _, improved = compare_eval_runs(baseline, candidate, [BadcaseCategory.RETRIEVAL_MISS])
    assert improved is False


def test_compare_empty_targets_only_checks_overall() -> None:
    baseline = _summary(grounding_check_rate=0.5, passed_cases=5)
    candidate = _summary(grounding_check_rate=0.5, passed_cases=5)
    _, improved = compare_eval_runs(baseline, candidate, [])
    assert improved is True


def test_build_eval_run_without_baseline() -> None:
    candidate = WorkflowEvalReport(summary=_summary(passed_cases=10))
    run = build_eval_run(
        candidate,
        None,
        model_version="aca-new",
        baseline_version="aca-old",
        dataset_ref="evals/cases.json",
        dataset_kind="golden",
        target_categories=[BadcaseCategory.RETRIEVAL_MISS],
    )
    assert run.model_version == "aca-new"
    assert run.improved is False  # no baseline -> cannot claim improvement
    assert run.delta_vs_baseline == {}


def test_eval_run_store_save_get_list_latest() -> None:
    store = EvalRunStore(":memory:")
    candidate = WorkflowEvalReport(summary=_summary(passed_cases=10))
    run = build_eval_run(
        candidate,
        None,
        model_version="aca-new",
        baseline_version="aca-old",
        dataset_ref="evals/cases.json",
        dataset_kind="golden",
        target_categories=[],
    )
    store.save(run)
    assert store.get(run.id) is not None
    assert store.latest_for_version("aca-new") is not None
    assert len(store.list_recent()) == 1


def test_rag_metrics_from_cases() -> None:
    cases = [
        RagEvalCase(query="q1", relevant_ids=["a", "b"], retrieved=["a", "c", "b"]),
        RagEvalCase(query="q2", relevant_ids=["x"], retrieved=["y", "x"]),
    ]
    metrics = rag_metrics_from_cases(cases, k=5)
    # q1 hits a at rank1 (mrr 1.0), q2 hits x at rank2 (mrr 0.5) -> avg 0.75
    assert metrics["mrr"] == 0.75
    assert metrics["hit_rate_at_5"] == 1.0
    assert metrics["ndcg_at_5"] > 0.5
