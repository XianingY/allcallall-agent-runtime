"""Tests for evaluation-gated routing and bounded independent-role execution."""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

import allcallall_agent_runtime.nodes.parallel_roles as parallel_roles
from allcallall_agent_runtime.config import AgentRuntimeConfig
from allcallall_agent_runtime.config import config as app_config
from allcallall_agent_runtime.dag import _route_after_parallel_roles, _route_first
from allcallall_agent_runtime.deadline import ExecutionCancelled
from allcallall_agent_runtime.eval_runner import evaluate_case, load_cases
from allcallall_agent_runtime.models import (
    ContextChunk,
    MeetingBriefRequest,
    MeetingBriefResponse,
    RoleResult,
    TraceEvent,
    WorkflowEvalCase,
)
from allcallall_agent_runtime.nodes.parallel_roles import (
    execute_parallel_roles,
    merge_role_deltas,
)
from allcallall_agent_runtime.nodes.role_router import (
    EarlyTerminationThresholds,
    route_roles,
    should_terminate_early,
)
from allcallall_agent_runtime.state import GraphState
from allcallall_agent_runtime.state import RoleAllocation


def _chunk(source_type: str, source_id: str = "1") -> ContextChunk:
    return ContextChunk(
        chunk_id=f"{source_type}-{source_id}",
        source_type=source_type,
        source_id=source_id,
        snippet=f"{source_type} evidence",
        score=10,
    )


def _request(preset: str = "meeting_brief") -> MeetingBriefRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=1,
        conversation_id=1,
        workflow_run_id=1,
        goal="Summarize the meeting and propose follow-ups.",
        preset=preset,
        context_chunks=[_chunk("meeting_transcript")],
    )


def _role_result(role: str) -> RoleResult:
    return RoleResult(role=role, summary=f"{role} result")


def _state() -> GraphState:
    return cast(
        GraphState,
        {
            "request": _request(),
            "provider": object(),
            "tool_bridge": object(),
            "trace_events": [TraceEvent(event="parent.trace", node="parent")],
            "role_results": [_role_result("old")],
            "checkpoint_writer": object(),
        },
    )


def _install_executors(
    monkeypatch: pytest.MonkeyPatch,
    searcher: Any,
    memory_agent: Any,
) -> None:
    monkeypatch.setattr(
        parallel_roles,
        "ROLE_EXECUTORS",
        {"searcher": searcher, "memory_agent": memory_agent},
    )


def test_new_performance_flags_default_off() -> None:
    cfg = AgentRuntimeConfig()
    assert cfg.enable_role_router is False
    assert cfg.enable_early_termination is False
    assert cfg.enable_parallel_roles is False


def test_early_termination_requires_all_quality_thresholds() -> None:
    thresholds = EarlyTerminationThresholds(0.8, 0.8, 0.8)
    accepted = should_terminate_early(
        evidence_sufficiency=0.8,
        citation_coverage=0.8,
        goal_coverage=0.8,
        required_roles_complete=True,
        unresolved_approval=False,
        safety_blocked=False,
        thresholds=thresholds,
    )
    assert accepted is True

    for field in (
        "evidence_sufficiency",
        "citation_coverage",
        "goal_coverage",
    ):
        values = {
            "evidence_sufficiency": 1.0,
            "citation_coverage": 1.0,
            "goal_coverage": 1.0,
        }
        values[field] = 0.79
        assert (
            should_terminate_early(
                required_roles_complete=True,
                unresolved_approval=False,
                safety_blocked=False,
                thresholds=thresholds,
                **values,
            )
            is False
        )


def test_risk_policy_prevents_early_termination() -> None:
    thresholds = EarlyTerminationThresholds(0.8, 0.8, 0.8)
    assert (
        should_terminate_early(
            evidence_sufficiency=1.0,
            citation_coverage=1.0,
            goal_coverage=1.0,
            required_roles_complete=True,
            unresolved_approval=True,
            safety_blocked=False,
            thresholds=thresholds,
        )
        is False
    )
    assert (
        should_terminate_early(
            evidence_sufficiency=1.0,
            citation_coverage=1.0,
            goal_coverage=1.0,
            required_roles_complete=True,
            unresolved_approval=False,
            safety_blocked=True,
            thresholds=thresholds,
        )
        is False
    )
    assert (
        should_terminate_early(
            evidence_sufficiency=1.0,
            citation_coverage=1.0,
            goal_coverage=1.0,
            required_roles_complete=False,
            unresolved_approval=False,
            safety_blocked=False,
            thresholds=thresholds,
        )
        is False
    )


def test_parallel_graph_routes_to_group_then_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: GraphState = {
        "role_allocation": RoleAllocation(
            roles=["searcher", "memory_agent", "synthesize", "risk_analyst"]
        )
    }
    monkeypatch.setattr(app_config, "enable_role_router", True)
    monkeypatch.setattr(app_config, "enable_parallel_roles", True)
    assert _route_first(state) == "parallel_roles"
    assert _route_after_parallel_roles(state) == "synthesize"

    monkeypatch.setattr(app_config, "enable_parallel_roles", False)
    assert _route_first(state) == "searcher"


def test_context_only_routing_skips_memory_and_risk() -> None:
    allocation = route_roles({"request": _request("context_qa")})["role_allocation"]
    assert allocation.roles == ["searcher", "synthesize"]
    assert {"memory_agent", "risk_analyst"}.issubset(allocation.skip_roles)


def test_risk_intent_write_policy_approval_and_safety_keep_risk_analyst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("context_qa")
    request.tool_policy.write_tools = ["write_conversation_message"]

    for state in (
        {"request": request, "intent_route": None},
        {"request": _request("context_qa"), "intent_route": None, "unresolved_approval": True},
        {"request": _request("context_qa"), "intent_route": None, "safety_blocked": True},
    ):
        allocation = route_roles(cast(GraphState, state))["role_allocation"]
        assert "risk_analyst" in allocation.roles
        assert "risk_analyst" in allocation.required_roles


def test_parallel_roles_use_branch_snapshots_and_merge_canonically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    original_request = state["request"]
    seen: list[GraphState] = []

    def searcher(branch: GraphState) -> dict[str, Any]:
        seen.append(branch)
        assert branch["request"] is not original_request
        assert "checkpoint_writer" not in branch
        assert branch["trace_events"] == []
        assert branch["role_results"] == []
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        seen.append(branch)
        branch["request"].context_chunks.append(_chunk("memory", "mutated"))
        branch["role_results"].append("must not leak")
        return {"role_results": [_role_result("memory_agent")]}

    _install_executors(monkeypatch, searcher, memory_agent)
    deltas = execute_parallel_roles(
        state,
        ["memory_agent", "searcher"],
        max_parallel=2,
        token_budget=10_000,
    )
    merged = merge_role_deltas(deltas, ["searcher", "memory_agent", "synthesize", "risk_analyst"])

    assert [delta["role"] for delta in deltas] == ["searcher", "memory_agent"]
    assert [result.role for result in merged["role_results"]] == ["searcher", "memory_agent"]
    assert merged["searcher"].role == "searcher"
    assert merged["memory_agent"].role == "memory_agent"
    assert len(original_request.context_chunks) == 1
    assert state["trace_events"] == [TraceEvent(event="parent.trace", node="parent")]
    assert state["role_results"] == [_role_result("old")]
    assert len(seen) == 2


def test_parallel_roles_overlap_only_searcher_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()
    both_started = threading.Barrier(2, timeout=2)

    def searcher(branch: GraphState) -> dict[str, Any]:
        del branch
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            both_started.wait()
        finally:
            with lock:
                active -= 1
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            both_started.wait()
        finally:
            with lock:
                active -= 1
        return {"role_results": [_role_result("memory_agent")]}

    _install_executors(monkeypatch, searcher, memory_agent)
    deltas = execute_parallel_roles(
        _state(),
        ["searcher", "memory_agent"],
        max_parallel=4,
        token_budget=10_000,
    )
    assert [delta["role"] for delta in deltas] == ["searcher", "memory_agent"]
    assert maximum == 2

    with pytest.raises(ValueError, match="only searcher and memory_agent"):
        execute_parallel_roles(
            _state(),
            ["searcher", "memory_agent", "synthesize"],
            max_parallel=2,
            token_budget=10_000,
        )


def test_parallel_failure_cancels_sibling_and_discards_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_cancelled = threading.Event()

    def searcher(branch: GraphState) -> dict[str, Any]:
        cancel_event = branch["branch_cancel_event"]
        assert cancel_event.wait(timeout=2)
        sibling_cancelled.set()
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        raise ValueError("memory branch failed")

    _install_executors(monkeypatch, searcher, memory_agent)
    with pytest.raises(ValueError, match="memory branch failed"):
        execute_parallel_roles(
            _state(),
            ["searcher", "memory_agent"],
            max_parallel=2,
            token_budget=10_000,
        )
    assert sibling_cancelled.is_set()


def test_parallel_cancellation_propagates_existing_deadline_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_cancelled = threading.Event()

    def searcher(branch: GraphState) -> dict[str, Any]:
        assert branch["branch_cancel_event"].wait(timeout=2)
        sibling_cancelled.set()
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        raise ExecutionCancelled("client_cancelled")

    _install_executors(monkeypatch, searcher, memory_agent)
    with pytest.raises(ExecutionCancelled):
        execute_parallel_roles(
            _state(),
            ["searcher", "memory_agent"],
            max_parallel=2,
            token_budget=10_000,
        )
    assert sibling_cancelled.is_set()


def test_budget_shortfall_forces_sequential_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def searcher(branch: GraphState) -> dict[str, Any]:
        del branch
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        threading.Event().wait(0.01)
        with lock:
            active -= 1
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        threading.Event().wait(0.01)
        with lock:
            active -= 1
        return {"role_results": [_role_result("memory_agent")]}

    _install_executors(monkeypatch, searcher, memory_agent)
    monkeypatch.setattr(parallel_roles, "estimate_role_tokens", lambda state, role: 60)
    deltas = execute_parallel_roles(
        _state(),
        ["searcher", "memory_agent"],
        max_parallel=2,
        token_budget=99,
    )
    assert [delta["role"] for delta in deltas] == ["searcher", "memory_agent"]
    assert maximum == 1


def test_parallel_reservation_is_released_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingReservation:
        def __init__(self) -> None:
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    reservation = RecordingReservation()
    monkeypatch.setattr(parallel_roles, "_reserve_group_budget", lambda roles, budget: reservation)
    monkeypatch.setattr(parallel_roles, "estimate_role_tokens", lambda state, role: 10)
    _install_executors(
        monkeypatch,
        lambda branch: {"role_results": [_role_result("searcher")]},
        lambda branch: {"role_results": [_role_result("memory_agent")]},
    )
    execute_parallel_roles(
        _state(),
        ["searcher", "memory_agent"],
        max_parallel=2,
        token_budget=100,
    )
    assert reservation.release_count == 1


def test_eval_fixture_and_runner_cover_role_routing() -> None:
    cases = {case.name: case for case in load_cases()}
    skippable = cases["context_qa_skips_memory_and_risk"]
    assert skippable.expected_selected_roles == ["searcher", "synthesize"]
    assert skippable.forbidden_selected_roles == ["memory_agent", "risk_analyst"]

    mandatory = cases["context_qa_write_policy_requires_risk"]
    assert "risk_analyst" in mandatory.expected_selected_roles

    case = WorkflowEvalCase(
        name="routing-contract",
        preset="context_qa",
        goal="answer",
        request=_request("context_qa"),
        expected_selected_roles=["searcher", "synthesize"],
        forbidden_selected_roles=["memory_agent", "risk_analyst"],
    )
    response = MeetingBriefResponse(
        prompt_version="test",
        trace_events=[TraceEvent(event="test", node="test")],
        role_results=[
            _role_result("searcher"),
            RoleResult(role="summarizer", summary="answer"),
        ],
    )
    result = evaluate_case(case, run_workflow=lambda request: response)
    assert result.role_routing_matched is True

    bad_response = response.model_copy(
        update={
            "role_results": [
                _role_result("searcher"),
                _role_result("memory_agent"),
                RoleResult(role="summarizer", summary="answer"),
            ]
        }
    )
    failed = evaluate_case(case, run_workflow=lambda request: bad_response)
    assert failed.role_routing_matched is False
