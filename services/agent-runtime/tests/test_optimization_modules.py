"""Tests for the implemented deep-optimization modules (Modules 1-7).

Every optimization is opt-in (gated behind a config flag that defaults to off),
so these tests assert both that the *new* capability works AND that the default
path is preserved (no behavior change when flags are off).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from allcallall_agent_runtime.config import config as app_config
from allcallall_agent_runtime.context_compression import SQLiteLongTermMemory
from allcallall_agent_runtime.dag import build_workflow_graph
from allcallall_agent_runtime.main import app
from allcallall_agent_runtime.mcp_tools import (
    EXEC_ASYNC,
    EXEC_SYNC,
    WRITE,
    MCPTool,
    MCPToolRegistry,
)
from allcallall_agent_runtime.models import (
    Citation,
    ContextChunk,
    InputAttachment,
    MeetingBriefRequest,
    OutputDecision,
    TerminationSignal,
    TerminationTrigger,
)
from allcallall_agent_runtime.tool_bridge import ToolObservation
from allcallall_agent_runtime.nodes.check import quality_check, safety_check
from allcallall_agent_runtime.nodes.role_router import (
    classify_complexity,
    next_role_after,
    route_roles,
)
from allcallall_agent_runtime.nodes.synthesis import (
    _compute_goal_achievement,
    _inline_checkagent_should_stop,
    bounded_react_search,
)
from allcallall_agent_runtime.state import GraphState, RoleAllocation
from allcallall_agent_runtime.tool_layer import StubGoToolBridge


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _chunk(source_type: str, source_id: str = "s1") -> ContextChunk:
    return ContextChunk(
        chunk_id=f"{source_type}-{source_id}",
        source_type=source_type,
        source_id=source_id,
        source_title=f"title-{source_type}",
        snippet=f"snippet for {source_type}",
        score=1,
    )


def _cite(source_type: str, source_id: str = "s1") -> Citation:
    return Citation(
        chunk_id=f"{source_type}-{source_id}",
        source_type=source_type,
        source_id=source_id,
        snippet=f"snippet for {source_type}",
    )


def _request(preset: str = "meeting_brief") -> MeetingBriefRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=1,
        conversation_id=1,
        workflow_run_id=1,
        goal="Summarize the meeting and propose follow-ups.",
        preset=preset,
    )


def _bridge_with(chunks: list[ContextChunk]) -> StubGoToolBridge:
    return StubGoToolBridge(
        default=ToolObservation(
            tool_name="query_context_chunks",
            input={},
            output_json="",
            chunks=tuple(chunks),
        )
    )


# --------------------------------------------------------------------------- #
# Module 1: ReAct determinable termination                                     #
# --------------------------------------------------------------------------- #

def test_default_off_attaches_signal_without_early_exit() -> None:
    """With early termination off, the loop still emits a TerminationSignal."""
    bridge = _bridge_with([_chunk("meeting_transcript")])
    result = bounded_react_search(
        request=_request(),
        role="searcher",
        max_iterations=3,
        tools=["query_context_chunks"],
        bridge=bridge,
        enable_early_termination=False,
    )
    assert isinstance(result.termination_signal, TerminationSignal)
    assert result.termination_signal.triggered
    # Default triggers only: the existing searcher citation exit or the hard cap.
    assert result.termination_signal.trigger in {
        TerminationTrigger.CITATION_SATISFIED,
        TerminationTrigger.MAX_ITERATIONS,
    }


def test_early_termination_goal_achieved_saves_iterations() -> None:
    """Opt-in goal achievement stops a non-searcher role before max_iterations."""
    # 5 chunks across 4 source types (incl. transcript + conversation) -> score ~0.9.
    chunks = [
        _chunk("meeting_transcript", "a"),
        _chunk("knowledge", "b"),
        _chunk("conversation", "c"),
        _chunk("message", "d"),
        _chunk("knowledge", "e"),
    ]
    bridge = _bridge_with(chunks)
    result = bounded_react_search(
        request=_request("risk_review"),
        role="risk_analyst",
        max_iterations=2,
        tools=["query_context_chunks", "query_recent_meetings"],
        bridge=bridge,
        enable_early_termination=True,
        goal_threshold=0.7,
    )
    sig = result.termination_signal
    assert sig is not None and sig.triggered
    assert sig.trigger == TerminationTrigger.GOAL_ACHIEVED
    assert sig.iterations_used == 1
    assert sig.iterations_saved >= 1
    assert sig.goal_score >= 0.7


def test_goal_achievement_empty_returns_zero() -> None:
    assert _compute_goal_achievement("searcher", _request(), [], [], 1) == 0.0


def test_inline_checkagent_sufficient_evidence() -> None:
    cites = [_cite("meeting_transcript"), _cite("knowledge"), _cite("conversation")]
    # meeting_brief requires a transcript; present -> sufficient.
    assert _inline_checkagent_should_stop("searcher", cites, _request("meeting_brief")) is True
    # Without a transcript for meeting_brief -> not sufficient.
    no_tx = [_cite("knowledge"), _cite("conversation"), _cite("message")]
    assert _inline_checkagent_should_stop("searcher", no_tx, _request("meeting_brief")) is False
    # Generic: 3+ diverse citations -> sufficient.
    assert _inline_checkagent_should_stop("searcher", cites, _request("react_general")) is True


def test_response_surfaces_termination_signals_unchanged_default() -> None:
    """Default workflow run still surfaces (empty) termination/decision fields."""
    from allcallall_agent_runtime.engineering_harness import EngineeringHarness

    resp = EngineeringHarness().run_workflow(_request())
    # Fields exist and default to safe values; no exception from new fields.
    assert isinstance(resp.termination_signals, list)
    assert resp.output_decision is None or isinstance(resp.output_decision, OutputDecision)


# --------------------------------------------------------------------------- #
# Module 2: Dynamic role allocation                                            #
# --------------------------------------------------------------------------- #

def test_classify_complexity_buckets() -> None:
    simple = _request()
    assert classify_complexity(simple) == "simple"

    # A long goal alone bumps the score to "moderate".
    moderate = _request()
    moderate.goal = "x" * 120
    assert classify_complexity(moderate) == "moderate"

    # risk_review preset + two attachments + a long goal -> score 3 -> "complex".
    complex_req = _request("risk_review")
    complex_req.attachments = [InputAttachment(), InputAttachment()]
    complex_req.goal = "x" * 120
    assert classify_complexity(complex_req) == "complex"


def test_route_roles_preserves_static_meeting_brief() -> None:
    alloc = route_roles({"request": _request("meeting_brief")})["role_allocation"]
    assert isinstance(alloc, RoleAllocation)
    assert alloc.roles == ["searcher", "memory_agent", "synthesize", "risk_analyst"]


def test_route_roles_context_qa_skips_risk_analyst() -> None:
    alloc = route_roles({"request": _request("context_qa")})["role_allocation"]
    assert alloc.roles == ["searcher", "synthesize"]
    assert "risk_analyst" in alloc.skip_roles


def test_next_role_after_chaining() -> None:
    state: GraphState = {"role_allocation": RoleAllocation(roles=["searcher", "synthesize"])}
    assert next_role_after(state, None) == "searcher"
    assert next_role_after(state, "searcher") == "synthesize"
    assert next_role_after(state, "synthesize") == "merge"
    # No allocation -> full canonical order.
    empty: GraphState = {}
    assert next_role_after(empty, None) == "searcher"
    assert next_role_after(empty, "risk_analyst") == "merge"


def test_dynamic_graph_compiles_when_role_router_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_config, "enable_role_router", True)
    try:
        graph = build_workflow_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "role_router" in node_names
    finally:
        monkeypatch.setattr(app_config, "enable_role_router", False)


# --------------------------------------------------------------------------- #
# Module 3: Two-tier CheckAgent OutputDecision                                 #
# --------------------------------------------------------------------------- #

def _critic(passed: bool, issues: list[str], sufficient: bool, coverage: float = 0.0) -> Any:
    from allcallall_agent_runtime.models import CriticResult

    return CriticResult(
        passed=passed,
        issues=issues,
        citation_coverage=coverage,
        budget_respected=True,
        context_sufficient=sufficient,
    )


def test_quality_check_accumulates_output_decision() -> None:
    out = quality_check({"critic_result": _critic(True, [], True, 0.8)})
    od = out["output_decision"]
    assert isinstance(od, OutputDecision)
    assert od.l1_decision == "pass"
    assert od.final_verdict == "accept"
    # Legacy keys preserved so existing projections keep working.
    assert out["last_check_decision"] == "pass"


def test_safety_check_accumulates_output_decision() -> None:
    out = safety_check({"proposed_tool_calls": [], "risk_flags": []})
    od = out["output_decision"]
    assert isinstance(od, OutputDecision)
    assert od.l2_decision == "pass"
    assert out["last_check_decision"] == "pass"


def test_quality_escalate_sets_verdict() -> None:
    out = quality_check({"critic_result": _critic(False, ["grounding_failed"], True)})
    od = out["output_decision"]
    assert od.l1_decision == "escalate"
    assert od.final_verdict == "escalate"


# --------------------------------------------------------------------------- #
# Module 4: SQLite long-term memory                                            #
# --------------------------------------------------------------------------- #

def test_sqlite_long_term_memory_put_retrieve() -> None:
    mem = SQLiteLongTermMemory(":memory:")
    try:
        mem.put("k1", "Alice owns API design")
        mem.put("k2", "Deadline is Oct 15")
        results = mem.retrieve("API design", top_k=5)
        assert len(results) >= 1
        assert "Alice" in results[0]
        # access count bumped on retrieval.
        assert mem.access_count("k1") >= 1
    finally:
        mem.close()


# --------------------------------------------------------------------------- #
# Module 5: Skill injection (opt-in, default off)                              #
# --------------------------------------------------------------------------- #

def test_skill_instructions_disabled_by_default() -> None:
    from allcallall_agent_runtime.harness import AllCallAllAgentHarness

    harness = AllCallAllAgentHarness()
    assert harness._resolve_skill_instructions(_request(), []) == ""


# --------------------------------------------------------------------------- #
# Module 6: MCP tool schema validation                                         #
# --------------------------------------------------------------------------- #

def test_mcp_register_valid_tool_passes() -> None:
    reg = MCPToolRegistry()
    reg.register(
        MCPTool(
            name="ok_tool",
            title="Ok",
            description="d",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
    )
    assert reg.get("ok_tool") is not None


def test_mcp_register_rejects_missing_required_property() -> None:
    reg = MCPToolRegistry()
    with pytest.raises(ValueError):
        reg.register(
            MCPTool(
                name="bad_tool",
                title="Bad",
                description="d",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["missing"]},
            )
        )


def test_mcp_register_rejects_sync_write() -> None:
    reg = MCPToolRegistry()
    with pytest.raises(ValueError):
        reg.register(
            MCPTool(
                name="sync_write",
                title="Sync Write",
                description="d",
                input_schema={"type": "object", "properties": {}},
                kind=WRITE,
                execution_mode=EXEC_SYNC,
            )
        )


def test_mcp_valid_write_async_passes() -> None:
    reg = MCPToolRegistry()
    reg.register(
        MCPTool(
            name="async_write",
            title="Async Write",
            description="d",
            input_schema={"type": "object", "properties": {}},
            kind=WRITE,
            execution_mode=EXEC_ASYNC,
        )
    )
    assert reg.get("async_write") is not None


# --------------------------------------------------------------------------- #
# Module 7: Queue metrics endpoint                                            #
# --------------------------------------------------------------------------- #

def test_tool_queue_metrics_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/v1/tool-queue/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_enqueued" in body
    assert "avg_attempts_per_task" in body
    assert "dead_letter_ratio" in body
    assert isinstance(body["total_enqueued"], int)
