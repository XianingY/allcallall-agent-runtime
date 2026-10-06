"""Tests for evaluation-gated routing and bounded independent-role execution."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import allcallall_agent_runtime.nodes.parallel_roles as parallel_roles
from allcallall_agent_runtime.config import AgentRuntimeConfig
from allcallall_agent_runtime.config import config as app_config
from allcallall_agent_runtime.dag import _route_after_parallel_roles, _route_first
from allcallall_agent_runtime.deadline import ExecutionCancelled
from allcallall_agent_runtime.eval_runner import EvalMode, evaluate_case, load_cases, run_eval
from allcallall_agent_runtime.models import (
    ContextChunk,
    IntentRoute,
    MeetingBriefRequest,
    MeetingBriefResponse,
    RoleResult,
    TraceEvent,
    WorkflowEvalCase,
)
from allcallall_agent_runtime.nodes.context import collect_context, retrieval_planner
from allcallall_agent_runtime.nodes.parallel_roles import (
    execute_parallel_roles,
    merge_role_deltas,
    parallel_roles_node,
)
from allcallall_agent_runtime.nodes.retrieval import (
    build_evidence_pack,
    rerank_context,
    retrieve_context,
    retrieval_loop,
    sufficiency_gate,
)
from allcallall_agent_runtime.nodes.role_router import route_roles
from allcallall_agent_runtime.nodes.synthesis import (
    bounded_react_search,
    decompose,
    risk_analyst,
    searcher,
)
from allcallall_agent_runtime.nodes.role_router import (
    EarlyTerminationThresholds,
    should_terminate_early,
)
from allcallall_agent_runtime.state import GraphState
from allcallall_agent_runtime.state import RoleAllocation
from allcallall_agent_runtime.tool_bridge import ToolObservation
from allcallall_agent_runtime.tool_layer import StubGoToolBridge


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


def test_group_reservation_accounts_tokens_and_provider_calls() -> None:
    reservation = parallel_roles.GroupReservation(
        roles=("searcher", "memory_agent"),
        token_budget=100,
        provider_calls=2,
    )
    reservation.acquire("searcher", 40)
    assert reservation.used_tokens == 40
    assert reservation.used_provider_calls == 1
    reservation.acquire("memory_agent", 50)
    assert reservation.used_tokens == 90
    assert reservation.used_provider_calls == 2
    reservation.release_role("searcher")
    assert reservation.used_tokens == 50
    assert reservation.used_provider_calls == 1
    reservation.release()
    assert reservation.used_tokens == 0
    assert reservation.used_provider_calls == 0


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
        {"request": _request("context_qa"), "intent_route": IntentRoute(intent="risk")},
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


def test_parallel_failure_cleanup_is_bounded_when_sibling_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = threading.Event()
    sibling_finished = threading.Event()
    unblock_sibling = threading.Event()

    def searcher(branch: GraphState) -> dict[str, Any]:
        del branch
        sibling_started.set()
        # Deliberately ignore branch_cancel_event and block longer than the test bound.
        unblock_sibling.wait(timeout=2.0)
        sibling_finished.set()
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        assert sibling_started.wait(timeout=1.0)
        raise ValueError("memory branch failed")

    _install_executors(monkeypatch, searcher, memory_agent)
    monkeypatch.setattr(app_config, "cancellation_grace_seconds", 0.05)

    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="memory branch failed"):
            execute_parallel_roles(
                _state(),
                ["searcher", "memory_agent"],
                max_parallel=2,
                token_budget=10_000,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert not sibling_finished.is_set()
    finally:
        unblock_sibling.set()
        assert sibling_finished.wait(timeout=1.0)


def test_parallel_failure_preserves_parent_state_and_releases_reservation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    parent_trace = [TraceEvent(event="parent.trace", node="parent")]
    parent_results = [_role_result("old")]
    state["trace_events"] = parent_trace
    state["role_results"] = parent_results

    class RecordingReservation:
        release_count = 0

        def acquire(self, role: str, estimated_tokens: int) -> None:
            return None

        def release_role(self, role: str) -> None:
            return None

        def release(self) -> None:
            self.release_count += 1

    reservation = RecordingReservation()
    monkeypatch.setattr(parallel_roles, "_reserve_group_budget", lambda roles, budget: reservation)

    def searcher(branch: GraphState) -> dict[str, Any]:
        del branch
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        del branch
        raise ValueError("memory branch failed")

    _install_executors(monkeypatch, searcher, memory_agent)
    with pytest.raises(ValueError, match="memory branch failed"):
        execute_parallel_roles(
            state,
            ["searcher", "memory_agent"],
            max_parallel=2,
            token_budget=10_000,
        )

    assert state["trace_events"] == parent_trace
    assert state["role_results"] == parent_results
    assert reservation.release_count == 1


def test_parallel_failure_real_checkpoint_saver_has_no_partial_branch_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = StateGraph(GraphState)
    graph.add_node("parallel_roles", parallel_roles_node)
    graph.set_entry_point("parallel_roles")
    graph.add_edge("parallel_roles", END)
    compiled = graph.compile(checkpointer=MemorySaver())

    def searcher(branch: GraphState) -> dict[str, Any]:
        return {"role_results": [_role_result("searcher")]}

    def memory_agent(branch: GraphState) -> dict[str, Any]:
        raise ValueError("memory branch failed")

    _install_executors(monkeypatch, searcher, memory_agent)
    state = cast(
        GraphState,
        {
            "request": _request(),
            "trace_events": [TraceEvent(event="parent.trace", node="parent")],
            "role_results": [_role_result("old")],
            "role_allocation": RoleAllocation(
                roles=["searcher", "memory_agent", "synthesize", "risk_analyst"]
            ),
        },
    )
    config = {"configurable": {"thread_id": "parallel-failure"}}

    with pytest.raises(ValueError, match="memory branch failed"):
        compiled.invoke(state, config)

    snapshots = list(compiled.get_state_history(config))
    assert snapshots
    committed_roles = {
        result.role
        for snapshot in snapshots
        for result in snapshot.values.get("role_results", [])
    }
    assert committed_roles == {"old"}


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

        def acquire(self, role: str, estimated_tokens: int) -> None:
            return None

        def release_role(self, role: str) -> None:
            return None

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


def test_parallel_roles_node_preserves_parent_trace_and_role_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    parent_trace = [
        TraceEvent(event="graph.node.started", node="collect_context"),
        TraceEvent(event="graph.node.completed", node="collect_context"),
        TraceEvent(event="router.role_allocation", node="role_router"),
    ]
    state["trace_events"] = list(parent_trace)
    state["role_results"] = [_role_result("old")]
    state["role_allocation"] = RoleAllocation(
        roles=["searcher", "memory_agent", "synthesize", "risk_analyst"]
    )

    def fake_searcher(branch: GraphState) -> dict[str, Any]:
        return {
            "trace_events": [TraceEvent(event="searcher.completed", node="searcher")],
            "role_results": [_role_result("searcher")],
        }

    def fake_memory_agent(branch: GraphState) -> dict[str, Any]:
        return {
            "trace_events": [TraceEvent(event="memory_agent.completed", node="memory_agent")],
            "role_results": [_role_result("memory_agent")],
        }

    _install_executors(monkeypatch, fake_searcher, fake_memory_agent)
    result = parallel_roles_node(state)

    assert result["trace_events"][: len(parent_trace)] == parent_trace
    assert [event.event for event in result["trace_events"][len(parent_trace) :]] == [
        "searcher.completed",
        "memory_agent.completed",
    ]
    assert [result.role for result in result["role_results"]] == [
        "old",
        "searcher",
        "memory_agent",
    ]


def test_real_graph_preserves_audit_trace_through_parallel_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = StateGraph(GraphState)
    graph.add_node("collect_context", collect_context)
    graph.add_node("retrieval_planner", retrieval_planner)
    graph.add_node("retrieval_loop", retrieval_loop)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("evidence_pack", build_evidence_pack)
    graph.add_node("sufficiency_gate", sufficiency_gate)
    graph.add_node("decompose", decompose)
    graph.add_node("role_router", route_roles)
    graph.add_node("parallel_roles", parallel_roles_node)
    graph.set_entry_point("collect_context")
    graph.add_edge("collect_context", "retrieval_planner")
    graph.add_edge("retrieval_planner", "retrieval_loop")
    graph.add_edge("retrieval_loop", "retrieve_context")
    graph.add_edge("retrieve_context", "rerank_context")
    graph.add_edge("rerank_context", "evidence_pack")
    graph.add_edge("evidence_pack", "sufficiency_gate")
    graph.add_edge("sufficiency_gate", "decompose")
    graph.add_edge("decompose", "role_router")
    graph.add_edge("role_router", "parallel_roles")
    graph.add_edge("parallel_roles", END)
    compiled = graph.compile()

    def fake_searcher(branch: GraphState) -> dict[str, Any]:
        return {
            "trace_events": [TraceEvent(event="searcher.completed", node="searcher")],
            "role_results": [_role_result("searcher")],
        }

    def fake_memory_agent(branch: GraphState) -> dict[str, Any]:
        return {
            "trace_events": [TraceEvent(event="memory_agent.completed", node="memory_agent")],
            "role_results": [_role_result("memory_agent")],
        }

    _install_executors(monkeypatch, fake_searcher, fake_memory_agent)
    monkeypatch.setattr(app_config, "enable_role_router", True)
    monkeypatch.setattr(app_config, "enable_parallel_roles", True)

    state = _state()
    state["tool_bridge"] = StubGoToolBridge()
    result = compiled.invoke(state)

    trace_nodes = [event.node for event in result["trace_events"]]
    for expected_node in (
        "collect_context",
        "retrieval_planner",
        "retrieval_loop",
        "retrieve_context",
        "rerank_context",
        "evidence_pack",
        "sufficiency_gate",
        "decompose",
        "role_router",
        "searcher",
        "memory_agent",
    ):
        assert expected_node in trace_nodes


def test_bounded_react_search_checks_cancellation_before_tool_call() -> None:
    event = threading.Event()
    event.set()

    class NeverCalledBridge:
        calls = 0

        def configured(self) -> bool:
            return True

        def execute_read_tool(self, *args: object, **kwargs: object) -> ToolObservation | None:
            type(self).calls += 1
            return None

    bridge = NeverCalledBridge()
    with pytest.raises(ExecutionCancelled):
        bounded_react_search(
            request=_request(),
            role="searcher",
            max_iterations=2,
            tools=["query_context_chunks"],
            bridge=bridge,
            branch_cancel_event=event,
        )
    assert bridge.calls == 0


def test_bounded_react_search_checks_cancellation_between_iterations() -> None:
    event = threading.Event()

    class SetEventAfterFirstCallBridge:
        calls = 0

        def configured(self) -> bool:
            return True

        def execute_read_tool(self, *args: object, **kwargs: object) -> ToolObservation | None:
            type(self).calls += 1
            if self.calls == 1:
                event.set()
            return ToolObservation(
                tool_name="query_context_chunks",
                input={},
                output_json="",
            )

    bridge = SetEventAfterFirstCallBridge()
    with pytest.raises(ExecutionCancelled):
        bounded_react_search(
            request=_request(),
            role="searcher",
            max_iterations=3,
            tools=["query_context_chunks"],
            bridge=bridge,
            branch_cancel_event=event,
        )
    assert bridge.calls == 1


def test_searcher_derives_early_termination_gate_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_should_terminate_early(**kwargs: object) -> bool:
        captured.update(kwargs)
        return False

    monkeypatch.setattr(
        "allcallall_agent_runtime.nodes.synthesis.should_terminate_early",
        fake_should_terminate_early,
    )
    monkeypatch.setattr(app_config, "enable_early_termination", True)

    state = _state()
    state["unresolved_approval"] = True
    state["safety_blocked"] = True
    state["role_allocation"] = RoleAllocation(
        roles=["searcher", "memory_agent", "synthesize", "risk_analyst"],
        required_roles=set(),
    )
    state["role_results"] = [_role_result("searcher")]
    state["tool_bridge"] = StubGoToolBridge(
        default=ToolObservation(
            tool_name="query_context_chunks",
            input={},
            output_json="",
            chunks=(_chunk("meeting_transcript"),),
        )
    )

    searcher(state)

    assert captured["unresolved_approval"] is True
    assert captured["safety_blocked"] is True
    assert captured["required_roles_complete"] is True


def test_risk_analyst_derives_early_termination_gate_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_should_terminate_early(**kwargs: object) -> bool:
        captured.update(kwargs)
        return False

    monkeypatch.setattr(
        "allcallall_agent_runtime.nodes.synthesis.should_terminate_early",
        fake_should_terminate_early,
    )
    monkeypatch.setattr(app_config, "enable_early_termination", True)

    state = _state()
    state["unresolved_approval"] = False
    state["safety_blocked"] = False
    state["role_allocation"] = RoleAllocation(
        roles=["searcher", "memory_agent", "synthesize", "risk_analyst"],
        required_roles=set(),
    )
    state["role_results"] = [
        _role_result("searcher"),
        _role_result("memory_agent"),
        _role_result("summarizer"),
    ]
    state["tool_bridge"] = StubGoToolBridge(
        default=ToolObservation(
            tool_name="query_context_chunks",
            input={},
            output_json="",
            chunks=(_chunk("meeting_transcript"),),
        )
    )

    risk_analyst(state)

    assert captured["unresolved_approval"] is False
    assert captured["safety_blocked"] is False
    assert captured["required_roles_complete"] is True


def test_eval_runner_parallel_and_early_termination_modes_do_not_regress(
    tmp_path: Path,
) -> None:
    cases = {case.name: case for case in load_cases()}
    selected = [cases["meeting_brief_grounded_transcript"]]
    fixture = tmp_path / "eval-cases.json"
    fixture.write_text(
        f"[{selected[0].model_dump_json(by_alias=True)}]",
        encoding="utf-8",
    )

    baseline = run_eval(fixture, mode=EvalMode.BASELINE)
    parallel = run_eval(fixture, mode=EvalMode.PARALLEL_ROLES)
    early = run_eval(fixture, mode=EvalMode.EARLY_TERMINATION)

    assert baseline.summary.passed_cases == parallel.summary.passed_cases == early.summary.passed_cases
    assert baseline.summary.task_success_rate == parallel.summary.task_success_rate == early.summary.task_success_rate
    assert baseline.summary.citation_grounding_rate == parallel.summary.citation_grounding_rate == early.summary.citation_grounding_rate
