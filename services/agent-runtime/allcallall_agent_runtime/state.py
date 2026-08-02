"""State type for the LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from .models import (
    Citation,
    ContextChunk,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
    GraphExpansion,
    IntentRoute,
    MemoryReflection,
    OutputDecision,
    RetrievalPlan,
    RetrievalAttempt,
    RiskAssessment,
    RoleResult,
    ToolProposal,
    TraceEvent,
    WorkflowRequest,
)


@dataclass
class RoleAllocation:
    """Result of dynamic role routing (Module 2).

    Decides which agent roles execute for a request and how they may be grouped
    for parallel execution. ``None`` on the graph state means routing was not
    applied (the static chain ran), preserving legacy behavior.
    """

    roles: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    skip_roles: set[str] = field(default_factory=set)
    rationale: str = ""
    complexity: str = "simple"  # simple | moderate | complex


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
    critic_result: CriticResult
    # Two-tier CheckAgent loop engineering (quality_check L1 / safety_check L2)
    critic_retries: int
    last_check_decision: str
    check_log: list[dict[str, Any]]
    # --- Module 2: dynamic role allocation outcome --- #
    role_allocation: RoleAllocation | None
    # --- Module 3: aggregated two-tier review decision --- #
    output_decision: OutputDecision | None
    # --- Module 5: resolved skill system instructions (opt-in) --- #
    skill_instructions: str
    # --- Module 4: retrieved durable long-term memory (opt-in) --- #
    long_term_memory: list[str]
