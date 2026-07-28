"""Tests for the two-tier CheckAgent loop engineering (Module 2).

Covers the L1 quality gate (``quality_check``) and L2 safety gate
(``safety_check``), their deterministic three-state decisions, the conditional
edge routing, and that the compiled graph wires both check nodes with a
bounded loop-back to ``synthesize``.
"""

from __future__ import annotations

from allcallall_agent_runtime.dag import build_workflow_graph
from allcallall_agent_runtime.models import CriticResult, ToolProposal
from allcallall_agent_runtime.nodes.check import (
    CheckDecision,
    quality_check,
    route_quality,
    route_safety,
    safety_check,
)


def _critic(passed: bool, issues: list[str], context_sufficient: bool, coverage: float = 0.0) -> CriticResult:
    return CriticResult(
        passed=passed,
        issues=issues,
        citation_coverage=coverage,
        budget_respected=True,
        context_sufficient=context_sufficient,
    )


def test_quality_pass_when_critic_ok():
    state = {"critic_result": _critic(True, [], True, 0.8)}
    out = quality_check(state)
    assert out["last_check_decision"] == CheckDecision.PASS.value


def test_quality_revise_on_insufficient_context():
    state = {
        "critic_result": _critic(False, ["insufficient_context_guarded"], False),
        "critic_retries": 0,
    }
    out = quality_check(state)
    assert out["last_check_decision"] == CheckDecision.REVISE.value
    # Loop counter advanced so the loop is bounded.
    assert out["critic_retries"] == 1


def test_quality_escalate_on_grounding_failure():
    state = {"critic_result": _critic(False, ["grounding_failed"], True)}
    out = quality_check(state)
    assert out["last_check_decision"] == CheckDecision.ESCALATE.value


def test_quality_revise_on_generic_quality_issue():
    state = {"critic_result": _critic(False, ["citation_coverage_missing"], True), "critic_retries": 0}
    out = quality_check(state)
    assert out["last_check_decision"] == CheckDecision.REVISE.value


def test_quality_budget_exhausted_accepts_draft():
    # Default max_quality_retries = 1, so retries==1 already exhausts the budget.
    state = {
        "critic_result": _critic(False, ["insufficient_context_guarded"], False),
        "critic_retries": 1,
    }
    out = quality_check(state)
    assert out["last_check_decision"] == CheckDecision.PASS.value


def test_safety_pass_when_all_writes_gated_and_no_risk():
    state = {
        "proposed_tool_calls": [ToolProposal(approval_required=True, tool_name="x", arguments={})],
        "risk_flags": [],
    }
    out = safety_check(state)
    assert out["last_check_decision"] == CheckDecision.PASS.value


def test_safety_escalate_on_ungated_write():
    state = {
        "proposed_tool_calls": [ToolProposal(approval_required=False, tool_name="x", arguments={})],
        "risk_flags": [],
    }
    out = safety_check(state)
    assert out["last_check_decision"] == CheckDecision.ESCALATE.value


def test_safety_escalate_on_risk_flags():
    state = {"proposed_tool_calls": [], "risk_flags": ["privacy_leak"]}
    out = safety_check(state)
    assert out["last_check_decision"] == CheckDecision.ESCALATE.value


def test_route_quality_revise_loops_to_synthesize():
    assert route_quality({"last_check_decision": CheckDecision.REVISE.value}) == "revise"
    assert route_quality({"last_check_decision": CheckDecision.PASS.value}) == "safety"
    assert route_quality({"last_check_decision": CheckDecision.ESCALATE.value}) == "safety"


def test_route_safety_always_advances_to_approval():
    assert route_safety({}) == "approve"


def test_graph_wires_check_agents_and_loop_edge():
    # Compilation itself validates the conditional path_map targets exist.
    g = build_workflow_graph()
    graph = g.get_graph()
    node_names = set(graph.nodes.keys())
    assert {"quality_check", "safety_check"}.issubset(node_names)

    # The bounded loop edge (quality_check "revise" -> synthesize) is wired as a
    # conditional edge. Confirm the routing decision returns the loop target.
    assert route_quality({"last_check_decision": CheckDecision.REVISE.value}) == "revise"
    # And that a non-revise decision advances to the L2 safety gate.
    assert route_quality({"last_check_decision": CheckDecision.PASS.value}) == "safety"
