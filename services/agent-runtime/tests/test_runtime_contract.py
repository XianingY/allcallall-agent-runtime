from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from allcallall_agent_runtime.api.app import create_app
from allcallall_agent_runtime.checkpoint.store import SQLiteCheckpointStore
from allcallall_agent_runtime.main import app
from allcallall_agent_runtime.models import (
    ApprovalDecision,
    MeetingBriefRequest,
    PendingApproval,
    PendingApprovalTool,
    ToolProposal,
    WorkflowResponse,
    WorkflowResumeRequest,
)
from allcallall_agent_runtime.orchestration.harness import (
    AllCallAllAgentHarness,
    CheckpointVersionConflictError,
)


GO_INITIAL_REQUEST: dict[str, Any] = {
    "request_id": "req-1",
    "execution_id": "workflow:123",
    "expected_checkpoint_version": 0,
    "tool_capability": "builtin",
    "organization_id": 1,
    "user_id": 7,
    "conversation_id": 42,
    "workflow_run_id": 123,
    "preset": "meeting_brief",
    "goal": "Generate a durable meeting brief",
    "messages": [],
    "notes": [],
    "meeting_transcripts": [],
    "context_chunks": [],
    "tool_policy": {"read_tools": [], "write_tools": []},
    "max_iterations": {"searcher": 3},
    "agentic_rag": {"enabled": False},
    "context_manifest": {
        "selected": {"messages": 1},
        "truncated": [],
        "serialized_bytes": 32,
        "estimated_tokens": 8,
        "sql_statements": 5,
    },
}


def _go_resume_payload(agent: bool = False) -> dict[str, Any]:
    payload = {
        "request_id": "req-1",
        "execution_id": "agent:321" if agent else "workflow:123",
        "expected_checkpoint_version": 4,
        "tool_capability": "builtin",
        "organization_id": 1,
        "user_id": 7,
        "conversation_id": 42,
        "workflow_run_id": 123,
        "resume": {
            "approval_request_id": "workflow:123:approval",
            "decisions": [{"tool_call_id": "call-1", "decision": "approve"}],
        },
    }
    if agent:
        payload["agent_run_id"] = 321
        payload["workflow_run_id"] = 0
    return payload


def test_go_request_and_response_contract_round_trip() -> None:
    class ContractHarness:
        def run_meeting_brief(self, request: MeetingBriefRequest) -> WorkflowResponse:
            return self.run_workflow(request)

        def run_workflow(self, request: MeetingBriefRequest) -> WorkflowResponse:
            assert request.execution_id == "workflow:123"
            assert request.expected_checkpoint_version == 0
            assert request.tool_capability == "builtin"
            assert request.context_manifest is not None
            return WorkflowResponse(
                status="requires_action",
                execution_id=request.execution_id,
                checkpoint_id="checkpoint-1",
                checkpoint_version=4,
                proposed_tool_calls=[
                    ToolProposal(
                        tool_call_id="call-1",
                        tool_name="write_conversation_message",
                        arguments={"body": "approved"},
                    )
                ],
                pending_approval=PendingApproval(
                    approval_request_id="workflow:123:approval",
                    tools=[
                        PendingApprovalTool(
                            tool_call_id="call-1",
                            tool_name="write_conversation_message",
                            arguments={"body": "approved"},
                            arguments_sha256="0" * 64,
                        )
                    ],
                ),
            )

    with patch("allcallall_agent_runtime.api.routes.get_harness", return_value=ContractHarness()):
        response = TestClient(app).post(
            "/v1/workflows/meeting-brief/run",
            json=GO_INITIAL_REQUEST,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution_id"] == "workflow:123"
    assert body["checkpoint_id"] == "checkpoint-1"
    assert body["checkpoint_version"] == 4
    assert body["pending_approval"]["approval_request_id"] == "workflow:123:approval"
    assert body["proposed_tool_calls"][0]["tool_call_id"] == "call-1"


def test_workflow_and_agent_resume_endpoints_accept_go_contract() -> None:
    class ResumeHarness:
        def resume_workflow(self, request: Any) -> WorkflowResponse:
            assert request.execution_id == "workflow:123"
            return WorkflowResponse(
                status="ready",
                execution_id="workflow:123",
                checkpoint_id="checkpoint-2",
                checkpoint_version=5,
                approval_decisions=[
                    ApprovalDecision(tool_call_id="call-1", decision="approve")
                ],
            )

        def resume_agent(self, request: Any) -> WorkflowResponse:
            assert request.execution_id == "agent:321"
            return WorkflowResponse(
                status="ready",
                execution_id="agent:321",
                checkpoint_id="agent-checkpoint-2",
                checkpoint_version=5,
                approval_decisions=[
                    ApprovalDecision(tool_call_id="call-1", decision="approve")
                ],
            )

    with patch("allcallall_agent_runtime.api.routes.get_harness", return_value=ResumeHarness()):
        client = TestClient(create_app())
        workflow = client.post("/v1/workflows/meeting_brief/resume", json=_go_resume_payload())
        agent = client.post("/v1/agents/react/resume", json=_go_resume_payload(agent=True))

    assert workflow.status_code == 200, workflow.text
    assert workflow.json()["approval_decisions"] == [
        {"tool_call_id": "call-1", "decision": "approve"}
    ]
    assert agent.status_code == 200, agent.text
    assert agent.json()["checkpoint_version"] == 5


def test_checkpoint_version_conflict_returns_go_error_code() -> None:
    class ConflictHarness:
        def resume_workflow(self, request: Any) -> WorkflowResponse:
            raise CheckpointVersionConflictError("runtime checkpoint version mismatch")

    with patch("allcallall_agent_runtime.api.routes.get_harness", return_value=ConflictHarness()):
        response = TestClient(create_app()).post(
            "/v1/workflows/meeting_brief/resume",
            json=_go_resume_payload(),
        )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "checkpoint_version_conflict"


def test_harness_returns_durable_checkpoint_and_resumes_approval() -> None:
    request = MeetingBriefRequest(
        organization_id=1,
        user_id=7,
        conversation_id=42,
        workflow_run_id=123,
        execution_id="workflow:123",
        expected_checkpoint_version=0,
        preset="meeting_brief",
        goal="请生成会议复盘，关注风险和行动项。",
        meeting_transcripts=[
            {
                "id": 10,
                "recording_session_id": 20,
                "recording_file_id": 30,
                "start_ms": 1000,
                "end_ms": 5000,
                "text": "本次会议确认需要跟进安全审批，预算截止日期存在风险。",
            }
        ],
        context_chunks=[
            {
                "chunk_id": "10",
                "source_type": "meeting_transcript",
                "source_id": "10",
                "source_title": "Task Eval Meeting",
                "title": "Task Eval Meeting",
                "snippet": "本次会议确认需要跟进安全审批，预算截止日期存在风险。",
                "score": 10,
                "retrieval_mode": "rules",
                "recording_session_id": 20,
                "recording_file_id": 30,
                "transcript_segment_id": 10,
                "start_ms": 1000,
                "end_ms": 5000,
            }
        ],
    )
    harness = AllCallAllAgentHarness(checkpoint_store=SQLiteCheckpointStore(":memory:"))

    initial = harness.run_workflow(request)
    assert initial.execution_id == "workflow:123"
    assert initial.checkpoint_id
    assert initial.checkpoint_version > 0
    assert initial.status == "requires_action"
    assert initial.proposed_tool_calls
    assert initial.pending_approval is not None
    assert initial.pending_approval.tools[0].tool_call_id
    assert initial.pending_approval.tools[0].arguments_sha256

    decisions = [
        {"tool_call_id": tool.tool_call_id, "decision": "approve"}
        for tool in initial.pending_approval.tools
    ]
    decision_digest = hashlib.sha256(
        json.dumps(decisions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    resume_execution_id = (
        f"workflow:123:resume:{initial.checkpoint_version}:{decision_digest}"
    )
    resume_request = {
        "request_id": "req-1",
        "execution_id": resume_execution_id,
        "expected_checkpoint_version": initial.checkpoint_version,
        "tool_capability": "builtin",
        "organization_id": 1,
        "user_id": 7,
        "conversation_id": 42,
        "workflow_run_id": 123,
        "resume": {
            "approval_request_id": initial.pending_approval.approval_request_id,
            "decisions": decisions,
        },
    }
    resumed = harness.resume_workflow(WorkflowResumeRequest.model_validate(resume_request))

    assert resumed.status == "ready"
    assert resumed.execution_id == resume_execution_id
    assert resumed.pending_approval is None
    assert resumed.checkpoint_id
    assert resumed.checkpoint_version > initial.checkpoint_version
    assert resumed.approval_decisions == [
        ApprovalDecision(tool_call_id=tool.tool_call_id, decision="approve")
        for tool in initial.pending_approval.tools
    ]


def test_pending_arguments_digest_matches_go_canonical_json() -> None:
    harness = AllCallAllAgentHarness(checkpoint_store=SQLiteCheckpointStore(":memory:"))
    proposal = ToolProposal(
        tool_call_id="call-1",
        tool_name="write_conversation_message",
        arguments={"body": "安全 approval", "priority": 2},
    )

    pending = harness._pending_approval("workflow:123", [proposal])

    assert pending is not None
    assert (
        pending.tools[0].arguments_sha256
        == "2e118db9cb87402edd3f5ed74caf4c11b89e137b75e1a261d384e453e35a3323"
    )
