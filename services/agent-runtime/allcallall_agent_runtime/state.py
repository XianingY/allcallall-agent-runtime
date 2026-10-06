"""State type for the LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
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
    ApprovalDecision,
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
    required_roles: set[str] = field(default_factory=set)
    rationale: str = ""
    complexity: str = "simple"  # simple | moderate | complex


class GraphState(TypedDict, total=False):
    """State type for the LangGraph workflow.

    Request-scoped deadline and cancellation state is propagated via the
    module-level context variable (:func:`deadline.get_current_deadline`)
    rather than graph state keys, so it is naturally excluded from checkpoint
    serialization and a resumed run starts with a fresh deadline.
    """

    request: WorkflowRequest
    provider: Any  # LLMProvider
    tool_bridge: Any  # GoToolBridge
    rag_runtime: Any  # RAGRuntimeClient
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
    approval_decisions: list[ApprovalDecision]
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
    # --- Task 12: per-run retrieval cache (not serialized) --- #
    retrieval_cache: Any  # RunRetrievalCache
    # --- Task 13: branch-local cooperative cancellation (not merged) --- #
    branch_cancel_event: threading.Event
    unresolved_approval: bool
    safety_blocked: bool
