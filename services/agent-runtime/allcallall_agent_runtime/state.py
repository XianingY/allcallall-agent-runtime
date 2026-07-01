"""State type for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from .models import (
    Citation,
    ContextChunk,
    ContextSufficiency,
    EvidencePack,
    GraphExpansion,
    IntentRoute,
    MemoryReflection,
    RetrievalPlan,
    RetrievalAttempt,
    RiskAssessment,
    RoleResult,
    ToolProposal,
    TraceEvent,
    WorkflowRequest,
)


class GraphState(TypedDict, total=False):
    """State type for the LangGraph workflow."""

    request: WorkflowRequest
    provider: Any  # LLMProvider
    tool_bridge: Any  # GoToolBridge
    trace_events: list[TraceEvent]
    role_results: list[RoleResult]
    agentic_rag_enabled: bool
    intent_route: IntentRoute
    graph_expansion: GraphExpansion
    retrieval_plan: RetrievalPlan
    retrieval_attempts: list[RetrievalAttempt]
    agentic_context_chunks: list[ContextChunk]
    retrieved_context_chunks: list[ContextChunk]
    reranked_context_chunks: list[ContextChunk]
    evidence_pack: EvidencePack
    context_sufficiency: ContextSufficiency
    searcher: RoleResult
    memory_agent: RoleResult
    summarizer: RoleResult
    risk_analyst: RoleResult
    risk_assessment: RiskAssessment
    memory_reflection: MemoryReflection
    summary: str
    action_items: list[str]
    next_step: str
    risk_flags: list[str]
    citations: list[Citation]
    proposed_tool_calls: list[ToolProposal]
    prompt_version: str
    grounding_check_result: dict[str, Any]
