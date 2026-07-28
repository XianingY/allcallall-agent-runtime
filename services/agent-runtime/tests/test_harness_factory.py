"""Harness three-layer decoupling tests.

Proves the ``AllCallAllAgentHarness`` orchestrates a full workflow while having
its three concerns (scheduling, durability, tool execution) injected, so each
layer is independently testable and swappable.
"""

from __future__ import annotations

from allcallall_agent_runtime.checkpoint.store import NullCheckpointStore
from allcallall_agent_runtime.factory import build_agent_harness
from allcallall_agent_runtime.models import ContextChunk, MeetingBriefRequest
from allcallall_agent_runtime.providers.base import RulesProvider
from allcallall_agent_runtime.tool_bridge import ToolObservation
from allcallall_agent_runtime.tool_layer import StubToolLayer


def _request() -> MeetingBriefRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=4,
        goal="summarize the meeting and list action items",
        context_chunks=[ContextChunk(source_type="meeting", source_id="m1", snippet="Team discussed Q3 roadmap.")],
    )


def test_default_factory_builds_graph_without_durability() -> None:
    harness = build_agent_harness()
    assert harness._get_graph() is not None
    assert harness.checkpoint_store.kind == "none"
    assert harness.tool_layer is not None


def test_injected_dependencies_run_end_to_end() -> None:
    observation = ToolObservation(
        tool_name="query_context_chunks",
        input={},
        output_json='{"chunks": []}',
        chunks=(),
    )
    harness = build_agent_harness(
        checkpoint_store=NullCheckpointStore(),
        tool_layer=StubToolLayer(default=observation),
        provider=RulesProvider(),
    )
    response = harness.run_meeting_brief(_request())

    assert response.status in ("ready", "requires_action")
    assert isinstance(response.summary, str)
    assert response.provider == "rules"
    # trace must be recorded by the harness for every run
    assert isinstance(response.trace_events, list)
    # route decision is always projected
    assert response.route_decision is not None
    assert response.route_decision.route in {
        "CHAT",
        "CONSULT",
        "RISK",
        "FOLLOW_UP",
        "MEETING_RECAP",
    }


def test_react_preset_uses_injected_provider() -> None:
    observation = ToolObservation(
        tool_name="query_context_chunks",
        input={},
        output_json='{"chunks": []}',
        chunks=(),
    )
    harness = build_agent_harness(
        checkpoint_store=NullCheckpointStore(),
        tool_layer=StubToolLayer(default=observation),
        provider=RulesProvider(),
    )
    response = harness.run_react_agent(_request())
    assert response.status in ("ready", "requires_action")
    assert response.route_decision is not None
