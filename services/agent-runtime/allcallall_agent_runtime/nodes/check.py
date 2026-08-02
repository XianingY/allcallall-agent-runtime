"""Two-tier CheckAgent loop engineering.

``quality_check`` (L1) and ``safety_check`` (L2) are dedicated review agents that
inspect the draft produced by the role chain and decide, deterministically, one
of three outcomes:

* ``pass``     — the draft is good enough to continue (L1) or safe to act on (L2).
* ``revise``   — L1 only; send the draft back to ``synthesize`` for another pass.
* ``escalate`` — the draft is unsafe / ungrounded; route to human approval.

Decisions are rule-driven (not LLM-judged) to stay reproducible and resistant
to prompt injection, consistent with the rest of the runtime. Every decision is
recorded as a ``check.decision`` trace event so the loop is fully auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from ..config import config as app_config
from ..models import OutputDecision, TraceEvent
from ..state import GraphState


class CheckDecision(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    ESCALATE = "escalate"


@dataclass
class CheckOutcome:
    agent: str
    decision: CheckDecision
    rationale: str
    score: float = 0.0


def _record(state: GraphState, outcome: CheckOutcome) -> None:
    state.setdefault("trace_events", []).append(
        TraceEvent(
            event="check.decision",
            node=outcome.agent,
            status=outcome.decision.value,
            metadata={
                "agent": outcome.agent,
                "decision": outcome.decision.value,
                "rationale": outcome.rationale,
                "score": outcome.score,
            },
        )
    )


def _outcome_payload(agent: str, outcome: CheckOutcome) -> dict[str, Any]:
    return {
        "check_log": [asdict(outcome)],
        "last_check_decision": outcome.decision.value,
    }


def quality_check(state: GraphState) -> dict[str, Any]:
    """L1 review: is the synthesized draft good enough to proceed?

    Consumes the consolidated ``CriticResult`` produced by ``critic_check`` and
    decides one of three outcomes. The decision is rule-driven (not LLM-judged)
    to stay reproducible and resistant to prompt injection.
    """
    critic = state.get("critic_result")
    retries = int(state.get("critic_retries", 0) or 0)
    max_retries = max(0, int(app_config.max_quality_retries))

    # Budget guard: never loop forever. Once exhausted, accept the current draft.
    if retries >= max_retries:
        outcome = CheckOutcome(
            "quality_check", CheckDecision.PASS,
            "retry budget exhausted; accepting current draft", 0.5,
        )
        _record(state, outcome)
        return _outcome_payload("quality_check", outcome)

    if critic is None:
        outcome = CheckOutcome(
            "quality_check", CheckDecision.PASS,
            "no critic result available; accepting draft", 0.5,
        )
        _record(state, outcome)
        return _outcome_payload("quality_check", outcome)

    issues = list(getattr(critic, "issues", []) or [])
    if not getattr(critic, "passed", True):
        # Distinguish "needs more context" (loop back) from "unsafe" (escalate).
        if "insufficient_context_guarded" in issues or not getattr(critic, "context_sufficient", True):
            outcome = CheckOutcome(
                "quality_check", CheckDecision.REVISE,
                "context insufficient; re-synthesize with broader retrieval", 0.3,
            )
            _record(state, outcome)
            # Increment the live retry counter so the loop is bounded.
            return {
                "critic_retries": retries + 1,
                "output_decision": _accumulate_quality(state, outcome),
                **_outcome_payload("quality_check", outcome),
            }
        if "grounding_failed" in issues:
            outcome = CheckOutcome(
                "quality_check", CheckDecision.ESCALATE,
                "grounding check failed; escalate to human review", 0.0,
            )
            _record(state, outcome)
            return {
                "output_decision": _accumulate_quality(state, outcome),
                **_outcome_payload("quality_check", outcome),
            }
        outcome = CheckOutcome(
            "quality_check", CheckDecision.REVISE,
            "critic flagged quality issues; re-synthesize", 0.3,
        )
        _record(state, outcome)
        return {
            "critic_retries": retries + 1,
            "output_decision": _accumulate_quality(state, outcome),
            **_outcome_payload("quality_check", outcome),
        }

    outcome = CheckOutcome(
        "quality_check", CheckDecision.PASS,
        "draft quality acceptable", float(getattr(critic, "citation_coverage", 0.0) or 0.0),
    )
    _record(state, outcome)
    return {
        "output_decision": _accumulate_quality(state, outcome),
        **_outcome_payload("quality_check", outcome),
    }


def _accumulate_quality(state: GraphState, outcome: CheckOutcome) -> OutputDecision:
    """Fold the L1 decision into the running :class:`OutputDecision` (Module 3)."""
    od = state.get("output_decision") or OutputDecision()
    od.l1_decision = outcome.decision.value
    od.quality_trend.append(outcome.decision.value)
    od.confidence_trajectory.append(outcome.score)
    od.check_log.append(asdict(outcome))
    if outcome.decision == CheckDecision.REVISE:
        od.revision_count += 1
        od.total_review_cycles += 1
    elif outcome.decision == CheckDecision.ESCALATE:
        od.final_verdict = "escalate"
        od.rationale = outcome.rationale
    elif outcome.decision == CheckDecision.PASS:
        # Tentative; L2 safety may still override to escalate.
        od.final_verdict = "accept"
        od.rationale = outcome.rationale
    return od


def safety_check(state: GraphState) -> dict[str, Any]:
    """L2 review: is it safe to surface / act on the proposed writes?"""
    proposed = state.get("proposed_tool_calls", []) or []
    for item in proposed:
        # A write that is not gated behind explicit approval is a hard safety violation.
        if getattr(item, "approval_required", True) is False:
            outcome = CheckOutcome(
                "safety_check", CheckDecision.ESCALATE,
                "unsafe write proposal detected (not approval-gated); escalate", 0.0,
            )
            _record(state, outcome)
            return {
                "output_decision": _accumulate_safety(state, outcome),
                **_outcome_payload("safety_check", outcome),
            }

    risk_flags = state.get("risk_flags", []) or []
    if risk_flags:
        outcome = CheckOutcome(
            "safety_check", CheckDecision.ESCALATE,
            f"policy risk flags raised ({len(risk_flags)}); escalate to human", 0.2,
        )
        _record(state, outcome)
        return {
            "output_decision": _accumulate_safety(state, outcome),
            **_outcome_payload("safety_check", outcome),
        }

    outcome = CheckOutcome(
        "safety_check", CheckDecision.PASS, "no unsafe writes or risk flags", 1.0,
    )
    _record(state, outcome)
    return {
        "output_decision": _accumulate_safety(state, outcome),
        **_outcome_payload("safety_check", outcome),
    }


def _accumulate_safety(state: GraphState, outcome: CheckOutcome) -> OutputDecision:
    """Fold the L2 decision into the running :class:`OutputDecision` (Module 3)."""
    od = state.get("output_decision") or OutputDecision()
    od.l2_decision = outcome.decision.value
    od.check_log.append(asdict(outcome))
    if outcome.decision == CheckDecision.ESCALATE:
        od.final_verdict = "escalate"
        od.rationale = outcome.rationale
    elif outcome.decision == CheckDecision.PASS and od.final_verdict == "accept":
        # L2 confirms the tentative L1 accept.
        od.rationale = od.rationale or outcome.rationale
    return od


def route_quality(state: GraphState) -> str:
    """Conditional edge from ``quality_check``: loop back or advance to L2."""
    return "revise" if state.get("last_check_decision") == CheckDecision.REVISE.value else "safety"


def route_safety(state: GraphState) -> str:
    """Conditional edge from ``safety_check``: always advance to the approval gate."""
    return "approve"
