from __future__ import annotations

from typing import Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..admission import AdmissionController, AdmissionRejected
from ..api_auth import require_auth
from ..async_tool_queue import get_default_tool_queue
from ..config import config as runtime_config
from ..helpers import SUPPORTED_WORKFLOWS
from ..harness import HarnessTimeoutExceeded, get_harness
from ..metrics import registry, workflow_runs_total
from ..models import (
    AgentRunRequest,
    AgentRunResponse,
    MeetingBriefRequest,
    MeetingBriefResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from ..skill_registry import build_production_registry


_R = TypeVar("_R")

router = APIRouter()


def _get_admission(request: Request) -> AdmissionController | None:
    """Retrieve the admission controller from application state.

    Returns ``None`` when the lifespan has not been initialized (e.g. in
    lightweight tests that construct the app without entering the lifespan
    context manager), in which case admission control is bypassed.
    """
    return getattr(request.app.state, "admission", None)


def _run_with_admission(
    request: Request,
    run_func: Callable[..., _R],
    run_request: object,
) -> _R:
    """Run a workflow through admission control.

    Acquires an admission lease, runs the workflow, and releases the lease
    in ``finally``.  Maps admission rejection and timeout to HTTP errors.

    When the admission controller is not available (e.g. in tests that bypass
    the lifespan), the workflow runs without admission control.
    """
    admission = _get_admission(request)
    if admission is None:
        workflow_runs_total.inc()
        try:
            return run_func(run_request)
        except HarnessTimeoutExceeded:
            raise HTTPException(status_code=504, detail="Workflow run exceeded the request timeout") from None

    org_id = getattr(run_request, "organization_id", 0)
    try:
        lease = admission.acquire(organization_id=org_id)
    except AdmissionRejected as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "runtime_overloaded", "reason": exc.reason},
            headers={"Retry-After": str(int(exc.retry_after_seconds))},
        ) from None
    try:
        workflow_runs_total.inc()
        return run_func(run_request)
    except HarnessTimeoutExceeded:
        raise HTTPException(status_code=504, detail="Workflow run exceeded the request timeout") from None
    finally:
        lease.close()


def run_meeting_brief(request: MeetingBriefRequest) -> MeetingBriefResponse:
    """Run the meeting brief workflow."""
    workflow_runs_total.inc()
    try:
        return get_harness().run_meeting_brief(request)
    except HarnessTimeoutExceeded:
        raise HTTPException(status_code=504, detail="Workflow run exceeded the request timeout") from None


def run_react_agent(request: AgentRunRequest) -> AgentRunResponse:
    """Run the react agent workflow."""
    workflow_runs_total.inc()
    try:
        return get_harness().run_react_agent(request)
    except HarnessTimeoutExceeded:
        raise HTTPException(status_code=504, detail="Workflow run exceeded the request timeout") from None


def run_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """Run a workflow with the given request."""
    workflow_runs_total.inc()
    try:
        return get_harness().run_workflow(request)
    except HarnessTimeoutExceeded:
        raise HTTPException(status_code=504, detail="Workflow run exceeded the request timeout") from None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "python_langgraph"}


@router.get("/ready")
def ready() -> dict[str, object]:
    # Readiness reflects whether the process can serve requests and which
    # downstream integrations are wired. We intentionally do NOT ping the Go
    # tool bridge or RAG runtime here: a liveness/ready probe that depends on
    # downstream health can cause cascading failures and flap during incidents.
    return {
        "status": "ready",
        "provider": runtime_config.provider,
        "provider_strict": runtime_config.provider_strict,
        "tool_bridge_configured": bool(runtime_config.tool_bridge_base_url and runtime_config.tool_bridge_token),
        "rag_runtime_configured": bool(runtime_config.rag_runtime_base_url),
        "agentic_rag": runtime_config.enable_agentic_rag,
    }


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus text exposition of in-process counters."""
    return Response(content=registry.render_prometheus(), media_type="text/plain; version=0.0.4")


@router.get("/v1/workflows")
def workflows() -> dict[str, list[str]]:
    return {"workflows": sorted(SUPPORTED_WORKFLOWS)}


@router.get("/v1/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "runtime": "python_langgraph",
        "harness": "allcallall_v1",
        "agents": ["react_general", "searcher", "memory_agent", "summarizer", "risk_guardian"],
        "workflows": sorted(SUPPORTED_WORKFLOWS),
        "input_modalities": ["text", "image_metadata", "audio_transcript", "video_transcript"],
        "intent_routes": ["chat", "consult", "risk"],
        "loop_engineering": {
            "contract": ["LoopSpec", "LoopState", "LoopStep", "LoopBudget", "LoopStopReason", "LoopTrace"],
            "bounded_roles": {"searcher": 3, "risk_guardian": 2, "memory_agent": 1, "follow_up_planner": 2},
            "write_tools": "proposal_only",
        },
        "rag": ["dynamic_routing", "agentic_refinement", "knowledge_graph_expansion"],
        "memory": ["reflection", "approval_gated_upsert"],
        "write_tools": "proposal_only",
        "tool_queue": {
            "mode": "async_after_approval",
            "retry": "bounded",
            "dead_letter": True,
        },
    }


@router.post("/v1/agents/react/run", response_model=AgentRunResponse, dependencies=[Depends(require_auth)])
def react_run(request: AgentRunRequest, fastapi_request: Request) -> AgentRunResponse:
    return _run_with_admission(fastapi_request, run_react_agent, request)


@router.post("/v1/workflows/meeting-brief/run", dependencies=[Depends(require_auth)])
def meeting_brief(request: MeetingBriefRequest, fastapi_request: Request) -> MeetingBriefResponse:
    return _run_with_admission(fastapi_request, run_meeting_brief, request)


@router.post("/v1/workflows/{preset}/run", response_model=WorkflowResponse, dependencies=[Depends(require_auth)])
def workflow_run(preset: str, request: WorkflowRequest, fastapi_request: Request) -> WorkflowResponse:
    return _run_with_admission(fastapi_request, run_workflow, request.model_copy(update={"preset": preset}))


@router.get("/v1/tool-queue/status", dependencies=[Depends(require_auth)])
def tool_queue_status() -> dict[str, object]:
    """Operational view of the async write-tool queue (Module 6).

    Shows how many approved write proposals are queued, in flight, done, or
    dead-lettered. Enabling ``PY_AGENT_ENABLE_TOOL_QUEUE`` makes workflow runs
    enqueue their approved writes here; a background worker consumes them.
    """
    queue = get_default_tool_queue()
    tasks = queue.all()
    return {
        "enabled": runtime_config.enable_tool_queue,
        "queued": sum(1 for t in tasks if t.status == "queued"),
        "processing": sum(1 for t in tasks if t.status == "processing"),
        "done": sum(1 for t in tasks if t.status == "done"),
        "dead": sum(1 for t in tasks if t.status == "dead"),
        "total": len(tasks),
    }


@router.get("/v1/skills", dependencies=[Depends(require_auth)])
def list_skills() -> dict[str, object]:
    """Resolve the deployed skill set, applying the security overlay.

    This is the production call path for :class:`SkillRegistry` /
    :class:`SecurityOverlay`: high-risk skills are force-routed through the safety
    plan and marked approval-required. Returns an empty list when skills are
    disabled or no manifest is configured.
    """
    reg = build_production_registry(runtime_config.skill_manifest_path or None)
    skills: list[dict[str, object]] = []
    for skill in reg.all():
        try:
            resolved = reg.resolve(skill.name)
        except KeyError:
            continue
        skills.append(
            {
                "name": resolved.name,
                "risk_level": resolved.risk_level,
                "requires_approval": resolved.requires_approval,
                "tools": list(resolved.allowed_tools),
            }
        )
    return {"enabled": runtime_config.enable_skills, "skills": skills}


@router.get("/v1/tool-queue/metrics", dependencies=[Depends(require_auth)])
def tool_queue_metrics() -> dict[str, object]:
    """Operational metrics for the async write-tool queue (Module 6/7).

    Exposes aggregate health signals useful for dashboards and alerting: how
    many tasks are in flight, the average number of attempts per task (a proxy
    for retry pressure), and the dead-letter ratio (a proxy for permanent
    failures). Enabling ``PY_AGENT_ENABLE_TOOL_QUEUE`` makes workflow runs
    enqueue their approved writes here; a background worker consumes them.
    """
    queue = get_default_tool_queue()
    tasks = queue.all()
    total = len(tasks)
    avg_attempts = (sum(t.attempts for t in tasks) / total) if total else 0.0
    dead_ratio = (sum(1 for t in tasks if t.status == "dead") / total) if total else 0.0
    return {
        "enabled": runtime_config.enable_tool_queue,
        "total_enqueued": total,
        "avg_attempts_per_task": round(avg_attempts, 2),
        "dead_letter_ratio": round(dead_ratio, 4),
        "dead_letters": len(queue.dead_letters()),
    }
