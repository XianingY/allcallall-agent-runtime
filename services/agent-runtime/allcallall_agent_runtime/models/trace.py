from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .retrieval import Citation


class TraceEvent(BaseModel):
    event: str
    node: str
    role: str = ""
    status: str = "completed"
    iteration: int | None = None
    thought: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


LoopStopReason = Literal[
    "completed",
    "confidence_reached",
    "max_iterations",
    "insufficient_context",
    "tool_error",
    "no_tool_needed",
]


class LoopBudget(BaseModel):
    max_steps: int = 0
    used_steps: int = 0
    read_tool_calls: int = 0
    write_tool_proposals: int = 0


class LoopSpec(BaseModel):
    role: str
    objective: str = ""
    max_steps: int = 0
    allowed_tools: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


class LoopStep(BaseModel):
    iteration: int = 0
    role: str = ""
    thought_summary: str = ""
    selected_skill: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = 0
    stop_reason: LoopStopReason = "completed"
    budget_used: LoopBudget = Field(default_factory=LoopBudget)


class LoopTrace(BaseModel):
    role: str
    spec: LoopSpec
    steps: list[LoopStep] = Field(default_factory=list)
    stop_reason: LoopStopReason = "completed"
    completed: bool = True
    budget: LoopBudget = Field(default_factory=LoopBudget)
    termination_signal: TerminationSignal | None = None


class TerminationTrigger(str, Enum):
    """Deterministic reason a bounded ReAct loop stopped (Module 1).

    ``MAX_ITERATIONS`` is the hard backstop that always fires when no earlier
    signal triggers; the others are opt-in early-exit signals gated behind
    ``enable_early_termination`` so default behavior is unchanged.
    """

    GOAL_ACHIEVED = "goal_achieved"
    CONFIDENCE_PLATEAU = "confidence_plateau"
    CHECKAGENT_EARLY_STOP = "checkagent_early_stop"
    MAX_ITERATIONS = "max_iterations"
    CITATION_SATISFIED = "citation_satisfied"
    TOOL_ERROR = "tool_error"


class TerminationSignal(BaseModel):
    """Carries the reason, trigger, and metrics of a ReAct loop termination.

    Attached to every :class:`RoleResult` so downstream projection
    (``LoopTrace``) and eval can reason about *why* the loop stopped and how
    many iterations were saved versus the configured ``max_iterations``.
    """

    triggered: bool = False
    trigger: TerminationTrigger | None = None
    reason: str = ""
    goal_score: float = 0.0
    confidence_at_exit: float = 0.0
    confidence_history: list[float] = Field(default_factory=list)
    iterations_used: int = 0
    iterations_saved: int = 0
    citations_found: int = 0


class RouteDecision(BaseModel):
    route: Literal["CHAT", "CONSULT", "RISK", "FOLLOW_UP", "MEETING_RECAP"] = "CHAT"
    intent: str = ""
    target_workflow: str = ""
    confidence: float = 0
    rationale: str = ""
    retrieval_strategy: str = ""


class CriticResult(BaseModel):
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    citation_coverage: float = 0
    budget_respected: bool = True
    write_proposal_safe: bool = True
    grounding_passed: bool = True
    context_sufficient: bool = True


class AgentHarnessMetadata(BaseModel):
    name: str = "allcallall_v1"
    graph_name: str = "workflow_dag_with_bounded_loops"
    runtime: str = "python_langgraph"
    prompt_version: str = ""
    input_modalities: list[str] = Field(default_factory=list)


class RoleResult(BaseModel):
    role: str
    summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    next_step: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    react_trace: list[TraceEvent] = Field(default_factory=list)
    termination_signal: TerminationSignal | None = None


class OutputDecision(BaseModel):
    """Single source of truth for the two-tier CheckAgent review (Module 3).

    Aggregates the L1 quality gate (``quality_check``) and L2 safety gate
    (``safety_check``) outcomes into one auditable, serializable object that
    is attached to the :class:`WorkflowResponse` and persisted in checkpoints.
    All fields are Optional-with-default so old checkpoints deserialize cleanly.
    """

    final_verdict: Literal["accept", "reject", "escalate"] = "accept"
    l1_decision: str = ""  # PASS | REVISE | ESCALATE
    l2_decision: str = ""  # PASS | ESCALATE
    revision_count: int = 0
    quality_trend: list[str] = Field(default_factory=list)
    confidence_trajectory: list[float] = Field(default_factory=list)
    check_log: list[dict[str, Any]] = Field(default_factory=list)
    total_review_cycles: int = 1
    rationale: str = ""

