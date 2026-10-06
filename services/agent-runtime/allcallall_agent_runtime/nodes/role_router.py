"""Dynamic role allocation router (Module 2).

Selects which agent roles should execute for a request, based on the workflow
preset (strongest signal), the intent route, and request complexity. The result
is a *subsequence* of the canonical role order ``_ROLE_ORDER`` so the dynamic
DAG can route between roles with simple conditional edges.

When ``app_config.enable_role_router`` is False the DAG uses the original static
chain and this module is never invoked, so default behavior is unchanged. When
it is True, the router still preserves the legacy chain for ``meeting_brief``
(the default mapping includes every role in the original order), so opting in is
safe for the common case and only trims clearly-redundant roles for presets like
``context_qa``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..helpers import (
    WORKFLOW_CONTEXT_QA,
    WORKFLOW_FOLLOW_UP_PLANNER,
    WORKFLOW_MEETING_BRIEF,
    WORKFLOW_REACT_GENERAL,
    WORKFLOW_RISK_REVIEW,
)
from ..models import TraceEvent, WorkflowRequest
from ..deadline import get_current_deadline
from ..metrics import early_termination_decisions_total, role_router_selected_roles_total
from ..state import GraphState, RoleAllocation

# Canonical execution order. All preset role lists are subsequences of this so
# the router only ever advances forward (never jumps back).
_ROLE_ORDER: list[str] = ["searcher", "memory_agent", "synthesize", "risk_analyst"]
CANONICAL_ROLE_ORDER: tuple[str, ...] = (
    "searcher",
    "memory_agent",
    "synthesize",
    "risk_analyst",
)

# Default role set per preset. ``meeting_brief`` mirrors the original static
# chain exactly (searcher -> memory_agent -> synthesize -> risk_analyst).
_PRESET_ROLES: dict[str, list[str]] = {
    WORKFLOW_MEETING_BRIEF: ["searcher", "memory_agent", "synthesize", "risk_analyst"],
    WORKFLOW_REACT_GENERAL: ["searcher", "memory_agent", "synthesize"],
    WORKFLOW_RISK_REVIEW: ["searcher", "synthesize", "risk_analyst"],
    WORKFLOW_FOLLOW_UP_PLANNER: ["searcher", "memory_agent", "synthesize"],
    # context_qa never needs the risk analyst.
    WORKFLOW_CONTEXT_QA: ["searcher", "synthesize"],
}


def classify_complexity(request: WorkflowRequest) -> str:
    """Classify request complexity to tune role selection.

    simple   — short goal, no attachments / transcripts.
    moderate — one of: attachments, transcripts, or a long goal.
    complex  — several signals, or an explicit risk review.
    """
    score = 0
    if len(request.attachments) >= 2:
        score += 1
    if request.meeting_transcripts:
        score += 1
    if len(request.goal) > 100:
        score += 1
    if request.preset == WORKFLOW_RISK_REVIEW:
        score += 1
    if score >= 3:
        return "complex"
    if score >= 1:
        return "moderate"
    return "simple"


def _ordered_subset(roles: list[str]) -> list[str]:
    """Return ``roles`` reordered to follow ``_ROLE_ORDER`` (dedup, drop unknowns)."""
    wanted = set(roles)
    return [r for r in _ROLE_ORDER if r in wanted]


@dataclass(frozen=True, slots=True)
class EarlyTerminationThresholds:
    """Quality thresholds that must all pass before a run may stop early."""

    evidence_sufficiency: float
    citation_coverage: float
    goal_coverage: float


def should_terminate_early(
    *,
    evidence_sufficiency: float,
    citation_coverage: float,
    goal_coverage: float,
    required_roles_complete: bool,
    unresolved_approval: bool,
    safety_blocked: bool,
    thresholds: EarlyTerminationThresholds,
) -> bool:
    """Return True only when quality, safety, approval, and role gates pass.

    The three quality dimensions use an inclusive threshold. Any incomplete
    required role, unresolved approval, or safety blocker is a hard rejection;
    it cannot be compensated for by higher quality scores.
    """
    decision = (
        evidence_sufficiency >= thresholds.evidence_sufficiency
        and citation_coverage >= thresholds.citation_coverage
        and goal_coverage >= thresholds.goal_coverage
        and required_roles_complete
        and not unresolved_approval
        and not safety_blocked
    )
    early_termination_decisions_total.labels(
        decision="accepted" if decision else "rejected"
    ).inc()
    return decision


def _risk_is_mandatory(request: WorkflowRequest, state: GraphState) -> bool:
    intent = state.get("intent_route")
    return bool(
        request.preset == WORKFLOW_RISK_REVIEW
        or (intent is not None and intent.intent == "risk")
        or request.tool_policy.write_tools
        or state.get("unresolved_approval", False)
        or state.get("safety_blocked", False)
        or state.get("risk_flags")
    )


def _parallel_groups(roles: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    if "searcher" in roles and "memory_agent" in roles:
        groups.append(["searcher", "memory_agent"])
    groups.extend([role] for role in roles if role not in {"searcher", "memory_agent"})
    return groups


def route_roles(state: GraphState) -> dict[str, Any]:
    """LangGraph node: dynamically allocate roles for this run.

    Returns the running ``trace_events`` (extended with a router event) and the
    computed :class:`RoleAllocation`. The DAG's conditional edges read
    ``role_allocation.roles`` to decide which role node to visit next.
    """
    _check_cancelled()
    request = state["request"]
    intent = state.get("intent_route")
    preset = request.preset

    roles = list(_PRESET_ROLES.get(preset, _PRESET_ROLES[WORKFLOW_REACT_GENERAL]))
    complexity = classify_complexity(request)

    # Intent refinement (additive, never removes the meeting brief's full chain).
    if intent is not None and intent.intent == "risk" and "risk_analyst" not in roles:
        roles.append("risk_analyst")
    if (
        intent is not None
        and intent.intent == "chat"
        and preset in (WORKFLOW_REACT_GENERAL, WORKFLOW_FOLLOW_UP_PLANNER)
        and len(roles) > 3
    ):
        roles = [r for r in roles if r in {"searcher", "synthesize"}]

    roles = _ordered_subset(roles)
    if _risk_is_mandatory(request, state) and "risk_analyst" not in roles:
        roles.append("risk_analyst")
    all_role_names = set(_ROLE_ORDER)
    skip_roles = all_role_names - set(roles)
    required_roles = {"risk_analyst"} if _risk_is_mandatory(request, state) else set()

    for role in roles:
        role_router_selected_roles_total.labels(role=role).inc()

    allocation = RoleAllocation(
        roles=roles,
        parallel_groups=_parallel_groups(roles),
        skip_roles=skip_roles,
        required_roles=required_roles,
        rationale=(
            f"preset={preset}, "
            f"intent={intent.intent if intent is not None else 'none'}, "
            f"complexity={complexity}"
        ),
        complexity=complexity,
    )

    trace = state.get("trace_events", [])
    trace.append(
        TraceEvent(
            event="router.role_allocation",
            node="role_router",
            status="completed",
            metadata={
                "roles": roles,
                "skip_roles": sorted(skip_roles),
                "complexity": complexity,
            },
        )
    )

    return {"trace_events": trace, "role_allocation": allocation}


def next_role_after(state: GraphState, current_role: str | None) -> str:
    """Resolve the next graph node after ``current_role`` (or the first role).

    Returns a role node name, or ``"merge"`` when there are no further roles.
    Used by the DAG's conditional edges so the dynamic graph can skip redundant
    roles while still reaching ``merge`` for the downstream pipeline.
    """
    allocation = state.get("role_allocation")
    roles = allocation.roles if allocation is not None else list(_ROLE_ORDER)
    if not roles:
        return "merge"
    if current_role is None:
        return roles[0]
    if current_role in roles:
        idx = roles.index(current_role)
        if idx < len(roles) - 1:
            return roles[idx + 1]
    return "merge"



def _check_cancelled() -> None:
    """Raise ExecutionCancelled if the current execution deadline has been cancelled or expired."""
    deadline = get_current_deadline()
    if deadline is not None:
        deadline.raise_if_cancelled()
