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
    model_history: str = ""  # bounded history injected by context compression
    long_term_memory: list[str] = Field(default_factory=list)  # L2 retrieved durable memory
    retrieval_mode: str = ""  # go_context | rag_runtime | hybrid; empty defaults to hybrid
    context_fingerprint: str = ""  # SHA-256 prefix of context chunk keys; empty means unset
    corpus_version: str = ""  # Opaque version tag for the indexed corpus; empty means unset

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

