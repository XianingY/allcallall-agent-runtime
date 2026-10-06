from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .context import MeetingBriefRequest
from .retrieval import Citation, GraphExpansion, IntentRoute, KnowledgeGraphEdge, RetrievalAttempt, RetrievalPlan
from .tools import ToolProposal
from .trace import (
    AgentHarnessMetadata,
    CriticResult,
    LoopBudget,
    LoopTrace,
    OutputDecision,
    RoleResult,
    RouteDecision,
    TerminationSignal,
    TraceEvent,
)


class EvidencePack(BaseModel):
    selected_chunk_ids: list[str] = Field(default_factory=list)
    rejected_count: int = 0
    confidence: float = 0
    source_types: list[str] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    route_intent: str = ""
    coverage: float = 0
    graph_edges: list[KnowledgeGraphEdge] = Field(default_factory=list)


class ContextSufficiency(BaseModel):
    sufficient: bool = True
    confidence: float = 1
    reason: str = ""
    missing_info: list[str] = Field(default_factory=list)


class MemoryReflection(BaseModel):
    conversation_summary: str = ""
    key_insights: list[str] = Field(default_factory=list)
    risk_lessons: list[str] = Field(default_factory=list)
    reinforcement_queries: list[str] = Field(default_factory=list)
    memory_write_recommended: bool = False
    reason: str = ""


class RiskAssessment(BaseModel):
    severity: Literal["none", "low", "medium", "high"] = "none"
    categories: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    guardrails: list[str] = Field(default_factory=list)


class MeetingBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "requires_action", "failed"] = "requires_action"
    runtime: str = "python_langgraph"
    provider: str = "rules"
    summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    next_step: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    role_results: list[RoleResult] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    proposed_tool_calls: list[ToolProposal] = Field(default_factory=list)
    prompt_version: str = ""
    grounding_check_result: dict[str, Any] = Field(default_factory=dict)
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    context_sufficiency: ContextSufficiency = Field(default_factory=ContextSufficiency)
    intent_route: IntentRoute = Field(default_factory=IntentRoute)
    route_decision: RouteDecision = Field(default_factory=RouteDecision)
    critic_result: CriticResult = Field(default_factory=CriticResult)
    harness: AgentHarnessMetadata = Field(default_factory=AgentHarnessMetadata)
    loop_traces: list[LoopTrace] = Field(default_factory=list)
    stop_reason: str = "completed"
    budget: LoopBudget = Field(default_factory=LoopBudget)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    memory_reflection: MemoryReflection = Field(default_factory=MemoryReflection)
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    output_decision: OutputDecision | None = None
    termination_signals: list[TerminationSignal] = Field(default_factory=list)
    error: str = ""


WorkflowRequest = MeetingBriefRequest
WorkflowResponse = MeetingBriefResponse
AgentRunRequest = MeetingBriefRequest
AgentRunResponse = MeetingBriefResponse


class WorkflowEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    preset: str = "meeting_brief"
    goal: str
    request: WorkflowRequest
    expected_status: str = "requires_action"
    required_output_substrings: list[str] = Field(default_factory=list)
    required_citation_source_types: list[str] = Field(default_factory=list)
    required_tool_proposals: list[str] = Field(default_factory=list)
    forbidden_tool_proposals: list[str] = Field(default_factory=list)
    expected_selected_roles: list[str] = Field(default_factory=list)
    forbidden_selected_roles: list[str] = Field(default_factory=list)
    expected_route: str = ""
    requires_unsupported_claim_guard: bool = False


class WorkflowEvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    preset: str
    passed: bool
    status: str
    task_success: bool
    citation_grounded: bool
    tool_intent_matched: bool
    approval_safe: bool
    unsupported_claim_guarded: bool
    prompt_schema_valid: bool = True
    route_matched: bool = True
    role_routing_matched: bool = True
    loop_completed: bool = True
    stop_reason_valid: bool = True
    memory_reflection_precise: bool = True
    grounding_check_passed: bool = True
    retrieval_refinement_succeeded: bool = True
    citation_coverage_passed: bool = True
    max_iteration_compliant: bool = True
    unnecessary_tool_calls_avoided: bool = True
    errors: list[str] = Field(default_factory=list)


class WorkflowEvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int = 0
    passed_cases: int = 0
    task_success_rate: float = 0
    citation_grounding_rate: float = 0
    tool_intent_match_rate: float = 0
    approval_safety_rate: float = 0
    unsupported_claim_guard_rate: float = 0
    prompt_schema_valid_rate: float = 0
    route_accuracy: float = 0
    role_routing_match_rate: float = 0
    loop_completion_rate: float = 0
    stop_reason_valid_rate: float = 0
    memory_reflection_precision: float = 0
    grounding_check_rate: float = 0
    retrieval_refinement_success_rate: float = 0
    citation_coverage_rate: float = 0
    max_iteration_compliance_rate: float = 0
    unnecessary_tool_call_rate: float = 0


class WorkflowEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: str = "python_langgraph"
    provider: str = "rules"
    summary: WorkflowEvalSummary = Field(default_factory=WorkflowEvalSummary)
    cases: list[WorkflowEvalCaseResult] = Field(default_factory=list)


class BadcaseSource(str, Enum):
    AUTO_REVIEW = "auto_review"          # L1/L2 CheckAgent verdict
    AUTO_RUNTIME = "auto_runtime"        # failed/timeout/grounding
    SAMPLING_AUDIT = "sampling_audit"    # offline human audit
    USER_FEEDBACK = "user_feedback"      # Phase 2: user thumbs-down / dismiss


class BadcaseCategory(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"            # 检索漏召/错召
    HALLUCINATION = "hallucination"              # 引用不实/编造
    ROUTE_ERROR = "route_error"                  # 意图路由错误
    APPROVAL_BYPASS = "approval_bypass"          # 写工具绕过审批
    TIMEOUT = "timeout"                          # 超时
    REVIEW_REJECT = "review_reject"              # L1/L2 拒绝
    USER_DECLINE = "user_decline"                # 用户不采纳（Phase 2）
    UNSUPPORTED_MISHANDLE = "unsupported_mishandle"  # 超范围请求处理不当
    RUNTIME_ERROR = "runtime_error"              # 通用运行时失败（自动）


class BadcaseSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BadcaseRecord(BaseModel):
    """A captured failure sample for labeling and SFT reflow (Part 1).

    Stores the full request/response snapshot so a badcase can be replayed or
    converted into a training sample later. ``auto_signals`` records the
    deterministic judgment evidence; human-assigned fields (``label_*``) are
    filled during labeling and gate ``sft_eligible``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: str
    organization_id: int
    workflow_run_id: int
    source: BadcaseSource
    category: BadcaseCategory
    severity: BadcaseSeverity
    request: WorkflowRequest
    response: WorkflowResponse
    auto_signals: dict[str, Any] = Field(default_factory=dict)
    label_corrected_response: WorkflowResponse | None = None
    label_note: str = ""
    labeled_by: str = ""
    labeled_at: str | None = None
    status: Literal["open", "labeled", "approved", "training", "resolved"] = "open"
    sft_eligible: bool = False


class ChatMessage(BaseModel):
    """A single turn in an SFT chat sample (Part 2)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class SFTSample(BaseModel):
    """A labeled badcase converted into a supervised fine-tuning sample (Part 2).

    Standard ``messages`` format so the JSONL export can be fed directly to an
    external fine-tuning platform. ``tags`` carry the badcase category/severity
    so training can stratify or reweight by failure type.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    badcase_id: str
    model_version: str
    route: str
    system_prompt: str
    messages: list[ChatMessage]
    tags: list[str] = Field(default_factory=list)
    quality_score: float = 0
    created_at: str


class ModelVersion(BaseModel):
    """A trained model artifact produced from badcase SFT reflow (Part 3)."""

    model_config = ConfigDict(extra="forbid")

    version: str
    base_version: str = ""
    trained_from_badcase_ids: list[str] = Field(default_factory=list)
    artifact_uri: str = ""
    created_at: str


class EvalRun(BaseModel):
    """One online-evaluation run comparing a candidate model against a baseline (Part 3)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model_version: str
    baseline_version: str
    dataset_ref: str
    dataset_kind: Literal["golden", "live_sample"]
    metrics: WorkflowEvalSummary = Field(default_factory=WorkflowEvalSummary)
    rag_metrics: dict[str, float] = Field(default_factory=dict)
    delta_vs_baseline: dict[str, float] = Field(default_factory=dict)
    target_badcase_categories: list[BadcaseCategory] = Field(default_factory=list)
    improved: bool = False
    created_at: str
