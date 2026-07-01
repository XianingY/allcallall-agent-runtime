"""Shared data models for AllCallAll agent-runtime and rag-runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextChunk(BaseModel):
    """A chunk of context from a knowledge source, conversation, or meeting transcript."""

    model_config = ConfigDict(extra="allow")

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


class RetrievalAttempt(BaseModel):
    """A single retrieval attempt in an agentic retrieval workflow."""

    step: int
    query: str
    tool_name: str = ""
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


class RetrievalRoute(BaseModel):
    """Dynamic route selected for a user query before retrieval."""

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
    """A lightweight relationship inferred from retrieved evidence."""

    edge_id: str
    source: str
    relation: str
    target: str
    evidence_chunk_id: str = ""
    confidence: float = 0


class GraphExpansion(BaseModel):
    """Knowledge-graph expansion metadata for an agentic retrieval run."""

    enabled: bool = False
    query_terms: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdge] = Field(default_factory=list)


class EvidencePack(BaseModel):
    """A pack of evidence collected from retrieval attempts."""

    selected_chunk_ids: list[str] = Field(default_factory=list)
    rejected_count: int = 0
    confidence: float = 0
    source_types: list[str] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    citations: list[ContextChunk] = Field(default_factory=list)
    route_intent: str = ""
    coverage: float = 0
    graph_edges: list[KnowledgeGraphEdge] = Field(default_factory=list)


class ContextSufficiency(BaseModel):
    """Result of a context sufficiency check."""

    sufficient: bool = False
    confidence: float = 0
    reason: str = ""
    missing_info: list[str] = Field(default_factory=list)


LoopStopReason = Literal[
    "completed",
    "confidence_reached",
    "max_iterations",
    "insufficient_context",
    "tool_error",
    "no_tool_needed",
]


class LoopBudget(BaseModel):
    """Budget consumed by a bounded Agent loop."""

    max_steps: int = 0
    used_steps: int = 0
    read_tool_calls: int = 0
    write_tool_proposals: int = 0


class LoopSpec(BaseModel):
    """Execution contract for one bounded role loop."""

    role: str
    objective: str = ""
    max_steps: int = 0
    allowed_tools: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


class LoopStep(BaseModel):
    """One step in a bounded ReAct-style loop."""

    iteration: int = 0
    role: str = ""
    thought_summary: str = ""
    selected_skill: str = ""
    input_schema: dict[str, object] = Field(default_factory=dict)
    observation: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = 0
    stop_reason: LoopStopReason = "completed"
    budget_used: LoopBudget = Field(default_factory=LoopBudget)


class LoopTrace(BaseModel):
    """Structured trace projected from low-level LangGraph/tool events."""

    role: str
    spec: LoopSpec
    steps: list[LoopStep] = Field(default_factory=list)
    stop_reason: LoopStopReason = "completed"
    completed: bool = True
    budget: LoopBudget = Field(default_factory=LoopBudget)


class RouteDecision(BaseModel):
    """Supervisor-level route decision for a user task."""

    route: Literal["CHAT", "CONSULT", "RISK", "FOLLOW_UP", "MEETING_RECAP"] = "CHAT"
    intent: str = ""
    target_workflow: str = ""
    confidence: float = 0
    rationale: str = ""
    retrieval_strategy: str = ""


class CriticResult(BaseModel):
    """Post-loop critic result for grounding, budget, and write safety."""

    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    citation_coverage: float = 0
    budget_respected: bool = True
    write_proposal_safe: bool = True
    grounding_passed: bool = True
    context_sufficient: bool = True


class AgentHarnessMetadata(BaseModel):
    """Metadata for the runtime harness that executed the Agent request."""

    name: str = "allcallall_v1"
    graph_name: str = "workflow_dag_with_bounded_loops"
    runtime: str = "python_langgraph"
    prompt_version: str = ""
    input_modalities: list[str] = Field(default_factory=list)
