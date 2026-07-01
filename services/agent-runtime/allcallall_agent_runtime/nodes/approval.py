"""Tool proposal, approval gate, and finalization nodes."""

from __future__ import annotations

from typing import Any

from ..models import (
    ContextSufficiency,
    MemoryReflection,
    RiskAssessment,
    ToolProposal,
    TraceEvent,
    WorkflowRequest,
)
from ..helpers import (
    CREATE_FOLLOW_UP_TASK,
    UPSERT_MEMORY,
    WRITE_CONVERSATION_MESSAGE,
    WORKFLOW_CONTEXT_QA,
    WORKFLOW_FOLLOW_UP_PLANNER,
    WORKFLOW_RISK_REVIEW,
    runtime_subject_id,
)
from ..state import GraphState


def propose_tools(state: GraphState) -> GraphState:
    """Propose write tools based on workflow output."""
    request = state["request"]
    risk_assessment = state.get("risk_assessment", RiskAssessment())
    base: dict[str, Any] = {
        "conversation_id": request.conversation_id,
        "summary": state.get("summary", ""),
        "action_items": state.get("action_items", []),
        "next_step": state.get("next_step", ""),
        "risk_flags": state.get("risk_flags", []),
        "risk_assessment": risk_assessment.model_dump(),
    }
    reflection = state.get("memory_reflection", MemoryReflection())
    message_arguments = {
        **base,
        "citations": [citation.model_dump(exclude_none=True) for citation in state.get("citations", [])],
        "memory_reflection": reflection.model_dump(),
    }
    sufficiency = state.get("context_sufficiency", ContextSufficiency())
    proposals = (
        []
        if not sufficiency.sufficient
        else workflow_tool_proposals(request, base, message_arguments, reflection)
    )
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="propose_tools", status="running"))
    if not sufficiency.sufficient:
        trace.append(
            TraceEvent(
                event="tool.proposal.skipped",
                node="propose_tools",
                status="skipped",
                observation="context is insufficient; write-tool proposals are suppressed",
                metadata={"reason": sufficiency.reason, "missing_info": sufficiency.missing_info},
            )
        )
    for proposal in proposals:
        trace.append(
            TraceEvent(
                event="tool.proposed",
                node="propose_tools",
                tool_name=proposal.tool_name,
                metadata={"reason": proposal.reason, "approval_required": proposal.approval_required},
            )
        )
    trace.append(
        TraceEvent(
            event="graph.node.completed",
            node="propose_tools",
            status="completed",
            metadata={"proposed_tool_calls": len(proposals)},
        )
    )
    return {"trace_events": trace, "proposed_tool_calls": proposals}


def approval_gate(state: GraphState) -> GraphState:
    """Wait for human approval of proposed tool calls."""
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="approval_gate", status="running"))
    trace.append(
        TraceEvent(
            event="approval.wait",
            node="approval_gate",
            status="requires_action",
            metadata={"pending_tools": [item.tool_name for item in state.get("proposed_tool_calls", [])]},
        )
    )
    return {"trace_events": trace}


def finalize(state: GraphState) -> GraphState:
    """Finalize the workflow execution."""
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="finalize", status="running"))
    trace.append(TraceEvent(event="graph.node.completed", node="finalize", status="completed"))
    return {"trace_events": trace}


def workflow_tool_proposals(
    request: WorkflowRequest,
    base: dict[str, Any],
    message_arguments: dict[str, Any],
    reflection: MemoryReflection | None = None,
) -> list[ToolProposal]:
    """Generate tool proposals based on workflow preset."""
    if request.preset == WORKFLOW_CONTEXT_QA:
        return []
    subject = runtime_subject_id(request)
    priority = "high" if request.preset == WORKFLOW_RISK_REVIEW else "normal"
    rate_limit_key = f"org:{request.organization_id}:conversation:{request.conversation_id}"
    proposals = [
        ToolProposal(
            tool_name=WRITE_CONVERSATION_MESSAGE,
            arguments=message_arguments,
            reason=f"Write the grounded {request.preset} result back to the conversation after human approval.",
            idempotency_key=f"{subject}:write_conversation_message:{request.preset}",
            priority=priority,
            rate_limit_key=rate_limit_key,
        )
    ]
    if request.preset == WORKFLOW_FOLLOW_UP_PLANNER:
        proposals.append(
            ToolProposal(
                tool_name=CREATE_FOLLOW_UP_TASK,
                arguments={
                    "conversation_id": request.conversation_id,
                    "task_type": "send_message",
                    "next_step": base.get("next_step", "") or "Follow up on the meeting commitments.",
                },
                reason="Create a concrete follow-up task only after human approval.",
                idempotency_key=f"{subject}:create_follow_up_task",
                queue_name="agent_followups",
                rate_limit_key=rate_limit_key,
            )
        )
        memory_key = "follow_up_commitments"
    elif request.preset == WORKFLOW_RISK_REVIEW:
        memory_key = "open_risk_register"
    else:
        memory_key = "latest_meeting_brief"
    reflection = reflection or MemoryReflection()
    if reflection.memory_write_recommended:
        proposals.append(
            ToolProposal(
                tool_name=UPSERT_MEMORY,
                arguments={
                    **base,
                    "key": memory_key,
                    "reflection": reflection.model_dump(),
                },
                reason=f"Persist {request.preset} output as scoped Agent memory after approval.",
                idempotency_key=f"{subject}:upsert_conversation_memory:{memory_key}",
                queue_name="agent_memory",
                priority=priority,
                rate_limit_key=rate_limit_key,
            )
        )
    return proposals
