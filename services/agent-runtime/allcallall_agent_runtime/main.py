from __future__ import annotations

from fastapi import FastAPI

from .helpers import SUPPORTED_WORKFLOWS
from .harness import AllCallAllAgentHarness
from .models import (
    AgentRunRequest,
    AgentRunResponse,
    MeetingBriefRequest,
    MeetingBriefResponse,
    WorkflowRequest,
    WorkflowResponse,
)


def run_meeting_brief(request: MeetingBriefRequest) -> MeetingBriefResponse:
    """Run the meeting brief workflow."""
    return AllCallAllAgentHarness().run_meeting_brief(request)


def run_react_agent(request: AgentRunRequest) -> AgentRunResponse:
    """Run the react agent workflow."""
    return AllCallAllAgentHarness().run_react_agent(request)


def run_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """Run a workflow with the given request."""
    return AllCallAllAgentHarness().run_workflow(request)

app = FastAPI(title="AllCallAll Agent Runtime", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "python_langgraph"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/v1/workflows")
def workflows() -> dict[str, list[str]]:
    return {"workflows": sorted(SUPPORTED_WORKFLOWS)}


@app.get("/v1/capabilities")
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


@app.post("/v1/agents/react/run", response_model=AgentRunResponse)
def react_run(request: AgentRunRequest) -> AgentRunResponse:
    return run_react_agent(request)


@app.post("/v1/workflows/meeting-brief/run")
def meeting_brief(request: MeetingBriefRequest) -> MeetingBriefResponse:
    return run_meeting_brief(request)


@app.post("/v1/workflows/{preset}/run", response_model=WorkflowResponse)
def workflow_run(preset: str, request: WorkflowRequest) -> WorkflowResponse:
    return run_workflow(request.model_copy(update={"preset": preset}))
