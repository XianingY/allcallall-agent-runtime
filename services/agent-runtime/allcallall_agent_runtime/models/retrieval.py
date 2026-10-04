from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

