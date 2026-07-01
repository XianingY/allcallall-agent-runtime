from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationMessage(BaseModel):
    id: int = 0
    sender_id: int = 0
    body: str = ""
    created_at: str | None = None


class ConversationNote(BaseModel):
    id: int = 0
    author_id: int = 0
    body: str = ""
    created_at: str | None = None


class MeetingTranscriptSegment(BaseModel):
    id: int = 0
    recording_session_id: int = 0
    recording_file_id: int = 0
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    speaker: str = ""


class ContextChunk(BaseModel):
    chunk_id: str = ""
    source_type: str
    source_id: str
    source_title: str = ""
    title: str = ""
    snippet: str
    score: int = 0
    retrieval_mode: str = ""
    bm25_rank: int = 0
    vector_rank: int = 0
    rrf_score: float = 0
    bm25_score: float = 0
    vector_score: float = 0
    rerank_score: float = 0
    rerank_reason: str = ""
    final_rank: int = 0
    recording_session_id: int | None = None
    recording_file_id: int | None = None
    transcript_segment_id: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None


class InputAttachment(BaseModel):
    """Optional non-text input metadata preprocessed by the Go backend."""

    attachment_id: str = ""
    modality: Literal["text", "image", "audio", "video", "file"] = "file"
    mime_type: str = ""
    uri: str = ""
    description: str = ""
    extracted_text: str = ""


class ToolPolicy(BaseModel):
    read_tools: list[str] = Field(default_factory=list)
    write_tools: list[str] = Field(default_factory=list)


class AgenticRAGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_steps: int = 3
    allowed_source_types: list[str] = Field(
        default_factory=lambda: [
            "meeting_transcript",
            "knowledge",
            "conversation",
            "message",
            "note",
            "followup",
            "memory",
            "contact_profile",
        ]
    )
    min_confidence: float = 0.6


class MeetingBriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = ""
    organization_id: int
    user_id: int
    conversation_id: int
    agent_run_id: int = 0
    workflow_run_id: int
    preset: str = "meeting_brief"
    goal: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    notes: list[ConversationNote] = Field(default_factory=list)
    meeting_transcripts: list[MeetingTranscriptSegment] = Field(default_factory=list)
    context_chunks: list[ContextChunk] = Field(default_factory=list)
    attachments: list[InputAttachment] = Field(default_factory=list)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    max_iterations: dict[str, int] = Field(default_factory=dict)
    agentic_rag: AgenticRAGConfig = Field(default_factory=AgenticRAGConfig)

    @field_validator(
        "messages",
        "notes",
        "meeting_transcripts",
        "context_chunks",
        "attachments",
        mode="before",
    )
    @classmethod
    def none_to_list(cls, value: object) -> object:
        return [] if value is None else value

    @field_validator("max_iterations", mode="before")
    @classmethod
    def none_to_dict(cls, value: object) -> object:
        return {} if value is None else value


class RetrievalPlanStep(BaseModel):
    step: int
    query: str
    source_scope: str = "all"
    tool_name: str = "query_context_chunks"
    rationale: str = ""
    strategy: str = "adaptive"
    expanded_terms: list[str] = Field(default_factory=list)


class IntentRoute(BaseModel):
    intent: Literal["chat", "consult", "risk"] = "chat"
    target_workflow: str = ""
    confidence: float = 0
    rationale: str = ""
    required_source_types: list[str] = Field(default_factory=list)
    retrieval_strategy: Literal[
        "none",
        "single_pass",
        "adaptive",
        "graph_augmented",
        "multi_hop",
    ] = "adaptive"


class KnowledgeGraphEdge(BaseModel):
    edge_id: str
    source: str
    relation: str
    target: str
    evidence_chunk_id: str = ""
    confidence: float = 0


class GraphExpansion(BaseModel):
    enabled: bool = False
    query_terms: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdge] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    enabled: bool = False
    max_steps: int = 3
    min_confidence: float = 0.6
    steps: list[RetrievalPlanStep] = Field(default_factory=list)
    intent_route: IntentRoute = Field(default_factory=IntentRoute)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)


class RetrievalAttempt(BaseModel):
    step: int
    query: str
    tool_name: str
    source_scope: str = "all"
    hit_count: int = 0
    source_types: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    observation: str = ""
    refined: bool = False
    confidence: float = 0
    strategy: str = ""
    expanded_terms: list[str] = Field(default_factory=list)
    graph_edge_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str = ""
    source_type: str
    source_id: str
    source_title: str = ""
    title: str = ""
    snippet: str
    score: int = 0
    retrieval_mode: str = ""
    rerank_score: float = 0
    rerank_reason: str = ""
    final_rank: int = 0
    recording_session_id: int | None = None
    recording_file_id: int | None = None
    transcript_segment_id: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None


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


class ToolProposal(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    idempotency_key: str = ""
    approval_required: bool = True
    execution_mode: Literal["proposal_only", "async_after_approval"] = "async_after_approval"
    queue_name: str = "agent_writebacks"
    priority: Literal["low", "normal", "high"] = "normal"
    max_attempts: int = 3
    rate_limit_key: str = ""
    dead_letter_queue: str = "agent_writebacks_dead_letter"


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
