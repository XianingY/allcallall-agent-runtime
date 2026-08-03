"""Online evaluation: compare a candidate model against a baseline (Part 3).

Reuses the deterministic 15-dimension workflow assertions from ``eval_runner``
and the IR metrics from ``engineering_harness`` (which mirror Go
``rag_eval.go``). Because the provider is currently a ``rules`` placeholder with
no trainable weights, this module compares two already-computed eval reports
(baseline vs candidate) and records the result. Wiring a real fine-tuned model
is left to the training platform that consumes the SFT dataset from Part 2.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal

from .engineering_harness import RagEvalCase, RagEvalHarness
from .models import (
    BadcaseCategory,
    EvalRun,
    WorkflowEvalReport,
    WorkflowEvalSummary,
)

# Map a badcase category to the eval metric whose improvement validates the fix.
_CATEGORY_METRIC: dict[BadcaseCategory, str] = {
    BadcaseCategory.RETRIEVAL_MISS: "grounding_check_rate",
    BadcaseCategory.HALLUCINATION: "citation_grounding_rate",
    BadcaseCategory.ROUTE_ERROR: "route_accuracy",
    BadcaseCategory.APPROVAL_BYPASS: "approval_safety_rate",
    BadcaseCategory.UNSUPPORTED_MISHANDLE: "unsupported_claim_guard_rate",
    BadcaseCategory.REVIEW_REJECT: "approval_safety_rate",
    BadcaseCategory.TIMEOUT: "task_success_rate",
    BadcaseCategory.RUNTIME_ERROR: "task_success_rate",
    BadcaseCategory.USER_DECLINE: "task_success_rate",
}


def _metric_fields(summary: WorkflowEvalSummary) -> dict[str, float]:
    return {
        "task_success_rate": summary.task_success_rate,
        "citation_grounding_rate": summary.citation_grounding_rate,
        "tool_intent_match_rate": summary.tool_intent_match_rate,
        "approval_safety_rate": summary.approval_safety_rate,
        "unsupported_claim_guard_rate": summary.unsupported_claim_guard_rate,
        "route_accuracy": summary.route_accuracy,
        "loop_completion_rate": summary.loop_completion_rate,
        "grounding_check_rate": summary.grounding_check_rate,
        "retrieval_refinement_success_rate": summary.retrieval_refinement_success_rate,
        "citation_coverage_rate": summary.citation_coverage_rate,
        "max_iteration_compliance_rate": summary.max_iteration_compliance_rate,
        "unnecessary_tool_call_rate": summary.unnecessary_tool_call_rate,
        "passed_cases": float(summary.passed_cases),
        "total_cases": float(summary.total_cases),
    }


def compare_eval_runs(
    baseline: WorkflowEvalSummary,
    candidate: WorkflowEvalSummary,
    target_categories: list[BadcaseCategory],
) -> tuple[dict[str, float], bool]:
    """Compute per-metric deltas and whether the candidate improved.

    ``improved`` is True when every targeted category's metric did not regress
    and the overall pass rate did not regress (no silent breakage elsewhere).
    """
    base = _metric_fields(baseline)
    cand = _metric_fields(candidate)
    delta = {key: round(cand[key] - base[key], 6) for key in cand}
    target_ok = True
    for category in target_categories:
        metric = _CATEGORY_METRIC[category]
        if delta.get(metric, 0.0) < 0:
            target_ok = False
    base_pass_rate = base["passed_cases"] / max(1, base["total_cases"])
    cand_pass_rate = cand["passed_cases"] / max(1, cand["total_cases"])
    no_overall_regression = cand["passed_cases"] >= base["passed_cases"] and (
        cand_pass_rate >= base_pass_rate - 1e-9
    )
    improved = target_ok and no_overall_regression
    return delta, improved


def build_eval_run(
    candidate: WorkflowEvalReport,
    baseline: WorkflowEvalReport | None,
    *,
    model_version: str,
    baseline_version: str,
    dataset_ref: str,
    dataset_kind: Literal["golden", "live_sample"],
    target_categories: list[BadcaseCategory],
) -> EvalRun:
    """Assemble an EvalRun from a candidate report and an optional baseline."""
    delta: dict[str, float] = {}
    improved = False
    if baseline is not None:
        delta, improved = compare_eval_runs(baseline.summary, candidate.summary, target_categories)
    return EvalRun(
        id=uuid.uuid4().hex,
        model_version=model_version,
        baseline_version=baseline_version,
        dataset_ref=dataset_ref,
        dataset_kind=dataset_kind,
        metrics=candidate.summary,
        delta_vs_baseline=delta,
        target_badcase_categories=list(target_categories),
        improved=improved,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def rag_metrics_from_cases(cases: list[RagEvalCase], k: int = 5) -> dict[str, float]:
    """Compute IR metrics over a labeled retrieval dataset (mirrors Go ``rag_eval``)."""
    result = RagEvalHarness().evaluate(cases, k=k)
    return {
        "hit_rate_at_5": result.hit_rate_at_5,
        "mrr": result.mrr,
        "ndcg_at_5": result.ndcg_at_5,
    }


class EvalRunStore:
    """SQLite-backed store for EvalRuns (same idiom as BadcaseStore)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    baseline_version TEXT NOT NULL,
                    dataset_ref TEXT NOT NULL,
                    dataset_kind TEXT NOT NULL,
                    improved INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def save(self, run: EvalRun) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO eval_runs "
                "(id, created_at, model_version, baseline_version, dataset_ref, dataset_kind, "
                "improved, payload) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run.id,
                    run.created_at,
                    run.model_version,
                    run.baseline_version,
                    run.dataset_ref,
                    run.dataset_kind,
                    int(run.improved),
                    run.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get(self, run_id: str) -> EvalRun | None:
        row = self._conn.execute("SELECT payload FROM eval_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return EvalRun.model_validate_json(row["payload"])

    def list_recent(self, limit: int = 100) -> list[EvalRun]:
        rows = self._conn.execute(
            "SELECT payload FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [EvalRun.model_validate_json(r["payload"]) for r in rows]

    def latest_for_version(self, model_version: str) -> EvalRun | None:
        row = self._conn.execute(
            "SELECT payload FROM eval_runs WHERE model_version=? ORDER BY created_at DESC LIMIT 1",
            (model_version,),
        ).fetchone()
        if row is None:
            return None
        return EvalRun.model_validate_json(row["payload"])
