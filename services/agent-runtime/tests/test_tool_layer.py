"""Tool layer decoupling tests.

Confirms the harness obtains its tool bridge through a ``ToolLayer`` protocol,
so production uses the real Go bridge while tests use a deterministic stub —
without any change to workflow node code.
"""

from __future__ import annotations

from allcallall_agent_runtime.models import (
    ContextChunk,
    MeetingBriefRequest,
    WorkflowRequest,
)
from allcallall_agent_runtime.tool_bridge import GoToolBridge, ToolObservation
from allcallall_agent_runtime.tool_layer import (
    GoToolBridgeLayer,
    StubGoToolBridge,
    StubToolLayer,
)


def _request() -> WorkflowRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=4,
        goal="summarize the meeting",
        context_chunks=[ContextChunk(source_type="meeting", source_id="m1", snippet="hello")],
    )


def test_go_layer_builds_real_bridge() -> None:
    bridge = GoToolBridgeLayer().build()
    assert isinstance(bridge, GoToolBridge)


def test_stub_layer_returns_canned_observation() -> None:
    obs = ToolObservation(tool_name="query_context_chunks", input={}, output_json="{}", chunks=())
    layer = StubToolLayer(observations={("query_context_chunks", "{}"): obs})
    bridge = layer.build()
    assert isinstance(bridge, StubGoToolBridge)
    assert bridge.execute_read_tool(_request(), "query_context_chunks", {}) is obs


def test_stub_layer_falls_back_to_default() -> None:
    obs = ToolObservation(tool_name="x", input={}, output_json="{}")
    layer = StubToolLayer(default=obs)
    assert layer.build().execute_read_tool(_request(), "unknown_tool", {"a": 1}) is obs


def test_stub_bridge_is_configured() -> None:
    bridge = StubToolLayer().build()
    assert bridge.configured() is True
