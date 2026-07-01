from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from shared.models import (
    AgentHarnessMetadata,
    ContextChunk,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
    GraphExpansion,
    LoopBudget,
    LoopTrace,
    RetrievalAttempt,
    RetrievalRoute,
    RouteDecision,
)


class ToolPolicy(BaseModel):
    read_tools: list[str] = Field(default_factory=list)
    write_tools: list[str] = Field(default_factory=list)


class AgenticRAGConfig(BaseModel):
    enabled: bool = False
    max_steps: int = 3
    allowed_source_types: list[str] = Field(default_factory=list)
    min_confidence: float = 0.6


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


class InputAttachment(BaseModel):
    attachment_id: str = ""
    modality: Literal["text", "image", "audio", "video", "file"] = "file"
    filename: str = ""
    mime_type: str = ""
    size_bytes: int = 0
    uri: str = ""
    description: str = ""
    extracted_text: str = ""
    ocr_text: str = ""
    caption_text: str = ""
    transcript_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = ""
    organization_id: int
    user_id: int
    conversation_id: int
    agent_run_id: int = 0
    workflow_run_id: int = 0
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


AgentRunRequest = WorkflowRequest


class Citation(ContextChunk):
    pass


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


class WorkflowResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ready", "requires_action", "failed"] = "requires_action"
    runtime: str = "python_langgraph"
    provider: str = "rules"
    summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    next_step: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    proposed_tool_calls: list[ToolProposal] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    context_sufficiency: ContextSufficiency = Field(default_factory=ContextSufficiency)
    intent_route: RetrievalRoute = Field(default_factory=RetrievalRoute)
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


AgentRunResponse = WorkflowResponse


class RetrievalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: int = 0
    user_id: int = 0
    conversation_id: int = 0
    query: str
    source_types: list[str] = Field(default_factory=list)
    top_k: int = 8
    query_vector: list[float] = Field(default_factory=list)
    chunks: list[ContextChunk] = Field(default_factory=list)


class RetrievalQueryResponse(BaseModel):
    runtime: str = "python_rag"
    query: str
    chunks: list[ContextChunk] = Field(default_factory=list)
    count: int = 0
    source: str = "inline"


class RerankRequest(BaseModel):
    query: str
    chunks: list[ContextChunk] = Field(default_factory=list)
    top_k: int = 8


class RerankResponse(BaseModel):
    runtime: str = "python_rag"
    provider: str = "rules"
    chunks: list[ContextChunk] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class AgenticRetrievalRequest(RetrievalQueryRequest):
    max_steps: int = 3
    min_confidence: float = 0.6


class AgenticRetrievalResponse(BaseModel):
    runtime: str = "python_rag"
    route: RetrievalRoute = Field(default_factory=RetrievalRoute)
    retrieval_route: RetrievalRoute = Field(default_factory=RetrievalRoute)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    attempts: list[RetrievalAttempt] = Field(default_factory=list)
    raw_hits: list[ContextChunk] = Field(default_factory=list)
    reranked_hits: list[ContextChunk] = Field(default_factory=list)
    rejected_chunks: list[ContextChunk] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    context_sufficiency: ContextSufficiency = Field(default_factory=ContextSufficiency)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    vector_store: str = "inline"


class GroundingCheckRequest(BaseModel):
    answer: str
    citations: list[ContextChunk] = Field(default_factory=list)


class GroundingCheckResponse(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    coverage: float = 0
    trace: dict[str, Any] = Field(default_factory=dict)
