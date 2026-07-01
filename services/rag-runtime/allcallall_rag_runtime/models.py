from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.models import (
    ContextChunk,
    ContextSufficiency,
    EvidencePack,
    GraphExpansion,
    KnowledgeGraphEdge,
    RetrievalAttempt,
    RetrievalRoute,
)

__all__ = [
    "ContextChunk",
    "ContextSufficiency",
    "EvidencePack",
    "GraphExpansion",
    "KnowledgeGraphEdge",
    "RetrievalAttempt",
    "RetrievalRoute",
]


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

    @field_validator("chunks", "source_types", "query_vector", mode="before")
    @classmethod
    def none_to_list(cls, value: object) -> object:
        return [] if value is None else value


class RetrievalQueryResponse(BaseModel):
    runtime: str = "python_rag"
    query: str
    chunks: list[ContextChunk] = Field(default_factory=list)
    count: int = 0
    source: str = "inline"


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[ContextChunk] = Field(default_factory=list)


class GroundingCheckResponse(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    coverage: float = 0
    trace: dict[str, Any] = Field(default_factory=dict)


class RAGEvalCase(BaseModel):
    name: str
    query: str
    chunks: list[ContextChunk]
    required_source_types: list[str] = Field(default_factory=list)
    expected_top_source_type: str = ""
    expected_route_intent: str = ""
    requires_graph_expansion: bool = False
    insufficient_context: bool = False


class RAGEvalCaseResult(BaseModel):
    name: str
    passed: bool
    top_source_type: str = ""
    route_intent: str = ""
    grounding_passed: bool = False
    sufficiency_passed: bool = False
    retrieval_refined: bool = False
    route_matched: bool = True
    graph_expanded: bool = True
    errors: list[str] = Field(default_factory=list)


class RAGEvalSummary(BaseModel):
    total_cases: int = 0
    passed_cases: int = 0
    rerank_top_match_rate: float = 0
    grounding_pass_rate: float = 0
    sufficiency_pass_rate: float = 0
    retrieval_refinement_success_rate: float = 0
    route_match_rate: float = 0
    graph_expansion_rate: float = 0


class RAGEvalReport(BaseModel):
    runtime: str = "python_rag"
    provider: str = "rules"
    summary: RAGEvalSummary = Field(default_factory=RAGEvalSummary)
    cases: list[RAGEvalCaseResult] = Field(default_factory=list)
