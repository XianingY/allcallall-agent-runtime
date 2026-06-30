from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from shared.models import ContextChunk, ContextSufficiency, EvidencePack, RetrievalAttempt


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
    attempts: list[RetrievalAttempt] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    context_sufficiency: ContextSufficiency = Field(default_factory=ContextSufficiency)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class GroundingCheckRequest(BaseModel):
    answer: str
    citations: list[ContextChunk] = Field(default_factory=list)


class GroundingCheckResponse(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    coverage: float = 0
    trace: dict[str, Any] = Field(default_factory=dict)

