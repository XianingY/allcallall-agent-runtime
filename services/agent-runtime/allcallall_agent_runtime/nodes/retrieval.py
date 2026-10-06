"""Retrieval loop and context processing nodes."""

from __future__ import annotations


from ..models import (
    ContextChunk,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
    RetrievalAttempt,
    RetrievalPlan,
    RetrievalPlanStep,
    TraceEvent,
)
from ..rag_runtime_client import RAGRuntimeClient, RAGRuntimeError
from ..retrieval import (
    RunRetrievalCache,
    compute_context_fingerprint,
    prepare_candidates,
    rerank_context_chunks,
    rerank_prepared,
)
from ..tool_bridge import ToolBridgeError
from ..helpers import (
    chunk_key,
    citations_from_chunks,
    dedupe_citations,
    estimate_retrieval_confidence,
    evaluate_context_sufficiency,
    first_non_empty,
    local_agentic_retrieval,
    summarize_observation,
    unique_strings,
)
from ..deadline import get_current_deadline
from ..state import GraphState
from ..synthesis import synthesize_action_items


# --------------------------------------------------------------------------- #
# Task 12: retrieval ownership helpers                                          #
# --------------------------------------------------------------------------- #


def _effective_retrieval_mode(request: object) -> str:
    """Return the effective retrieval mode, defaulting to ``hybrid``."""
    mode = getattr(request, "retrieval_mode", "") or ""
    if mode in ("go_context", "rag_runtime", "hybrid"):
        return mode
    return "hybrid"


def _resolve_fingerprint(request: object) -> str:
    """Return the context fingerprint from the request, or compute one."""
    fp = getattr(request, "context_fingerprint", "") or ""
    if fp:
        return fp
    chunks = getattr(request, "context_chunks", None) or []
    return compute_context_fingerprint(chunks)


def _resolve_corpus_version(request: object) -> str:
    return getattr(request, "corpus_version", "") or ""


def _should_call_rag(
    mode: str,
    plan: RetrievalPlan,
    step: RetrievalPlanStep,
    gathered: list[ContextChunk],
) -> bool:
    """Decide whether the current step should call the RAG runtime.

    In ``go_context`` mode, RAG is never called.
    In ``rag_runtime`` mode, RAG is always called.
    In ``hybrid`` mode, RAG is called only when context sufficiency is below
    threshold or the plan requires a source type absent from gathered chunks.
    """
    if mode == "go_context":
        return False
    if mode == "rag_runtime":
        return True
    # hybrid: call RAG when the step scope is not covered by gathered chunks
    # or when the plan explicitly requires a source type we don't have yet.
    gathered_types = {c.source_type for c in gathered}
    required = plan.intent_route.required_source_types or []
    if step.source_scope != "all" and step.source_scope not in gathered_types:
        return True
    missing = [t for t in required if t not in gathered_types]
    if missing:
        return True
    return False


# --------------------------------------------------------------------------- #
# Retrieval loop                                                                #
# --------------------------------------------------------------------------- #


def retrieval_loop(state: GraphState) -> GraphState:
    """Execute bounded retrieval loop with refinement."""
    request = state["request"]
    trace = state.get("trace_events", [])
    plan = state.get("retrieval_plan", RetrievalPlan())
    trace.append(TraceEvent(event="graph.node.started", node="retrieval_loop", status="running"))
    # Cooperative cancellation checkpoint before entering the retrieval loop.
    _check_cancelled()

    retrieval_mode = _effective_retrieval_mode(request)
    context_fingerprint = _resolve_fingerprint(request)
    corpus_version = _resolve_corpus_version(request)
    cache: RunRetrievalCache = state.get("retrieval_cache") or RunRetrievalCache()

    if not plan.enabled:
        trace.append(
            TraceEvent(
                event="rag.observe",
                node="retrieval_loop",
                status="skipped",
                observation="agentic rag disabled; using preloaded Go context",
                metadata={"preloaded_context_chunks": len(request.context_chunks)},
            )
        )
        if retrieval_mode == "go_context":
            trace.append(
                TraceEvent(
                    event="retrieval.mode",
                    node="retrieval_loop",
                    status="completed",
                    observation="go_context mode: using preloaded Go context only",
                    metadata={"retrieval_mode": "go_context", "context_chunks": len(request.context_chunks)},
                )
            )
        trace.append(TraceEvent(event="graph.node.completed", node="retrieval_loop", status="completed"))
        return {"trace_events": trace, "retrieval_attempts": [], "agentic_context_chunks": [], "retrieval_cache": cache}

    bridge = state["tool_bridge"]
    rag_runtime = state.get("rag_runtime") or RAGRuntimeClient()
    attempts: list[RetrievalAttempt] = []
    gathered: list[ContextChunk] = []
    seen_chunks: set[str] = set()
    confidence = 0.0

    for step in plan.steps[: max(1, min(plan.max_steps, 3))]:
        # Check cancellation at each bounded loop iteration.
        _check_cancelled()
        tool_input = {
            "conversation_id": request.conversation_id,
            "query": step.query,
            "limit": 6,
            "source_type": step.source_scope,
            "strategy": step.strategy,
            "expanded_terms": step.expanded_terms,
            "route_intent": plan.intent_route.intent,
        }
        trace.append(
            TraceEvent(
                event="rag.tool_call",
                node="retrieval_loop",
                status="running",
                iteration=step.step,
                tool_name=step.tool_name,
                tool_input=tool_input,
                metadata={"source_scope": step.source_scope, "rationale": step.rationale},
            )
        )
        selected: list[ContextChunk] = []
        observation_suffix = " via preloaded_context"
        used_runtime = False

        # --- Task 12: retrieval ownership and per-run cache --- #
        cached = cache.get(
            step.query, step.source_scope, plan.intent_route.retrieval_strategy,
            context_fingerprint, corpus_version,
        )
        if cached is not None:
            selected = cached
            observation_suffix = " via cache"
            trace.append(
                TraceEvent(
                    event="rag.cache_hit",
                    node="retrieval_loop",
                    status="completed",
                    iteration=step.step,
                    metadata={"query": step.query[:80], "source_scope": step.source_scope},
                )
            )
        elif _should_call_rag(retrieval_mode, plan, step, gathered):
            try:
                runtime_observation = rag_runtime.agentic_retrieve(request, step, plan)
                if runtime_observation is not None:
                    selected = list(runtime_observation.chunks)
                    used_runtime = True
                    observation_suffix = " via rag_runtime"
                    trace.append(
                        TraceEvent(
                            event="rag.runtime_call",
                            node="retrieval_loop",
                            status="completed",
                            iteration=step.step,
                            tool_name="rag_runtime.agentic",
                            metadata={
                                "confidence": runtime_observation.confidence,
                                "sufficient": runtime_observation.sufficient,
                                "attempts": runtime_observation.attempts,
                                "returned": len(selected),
                            },
                        )
                    )
            except RAGRuntimeError as exc:
                trace.append(
                    TraceEvent(
                        event="rag.runtime_call",
                        node="retrieval_loop",
                        status="failed",
                        iteration=step.step,
                        tool_name="rag_runtime.agentic",
                        observation=str(exc),
                        metadata={"fallback": "go_tool_bridge_or_preloaded_context"},
                    )
                )
            # RAG failed or returned nothing — fall through to local + tool bridge.
            if not selected:
                selected = local_agentic_retrieval(request.context_chunks, step)
            if not used_runtime:
                try:
                    bridge_observation = bridge.execute_read_tool(request, step.tool_name, tool_input)
                    if bridge_observation is not None:
                        selected = list(bridge_observation.chunks)
                        observation_suffix = " via go_tool_bridge"
                except ToolBridgeError as exc:
                    trace.append(
                        TraceEvent(
                            event="rag.tool_bridge_call",
                            node="retrieval_loop",
                            status="failed",
                            iteration=step.step,
                            tool_name=step.tool_name,
                            observation=str(exc),
                            metadata={"fallback": "preloaded_context"},
                        )
                    )
        else:
            # go_context or hybrid with sufficient context: use local chunks only.
            selected = local_agentic_retrieval(request.context_chunks, step)
            if not selected:
                try:
                    bridge_observation = bridge.execute_read_tool(request, step.tool_name, tool_input)
                    if bridge_observation is not None:
                        selected = list(bridge_observation.chunks)
                        observation_suffix = " via go_tool_bridge"
                except ToolBridgeError as exc:
                    trace.append(
                        TraceEvent(
                            event="rag.tool_bridge_call",
                            node="retrieval_loop",
                            status="failed",
                            iteration=step.step,
                            tool_name=step.tool_name,
                            observation=str(exc),
                            metadata={"fallback": "preloaded_context"},
                        )
                    )
            trace.append(
                TraceEvent(
                    event="rag.local_only",
                    node="retrieval_loop",
                    iteration=step.step,
                    status="completed",
                    metadata={"retrieval_mode": retrieval_mode, "source_scope": step.source_scope},
                )
            )

        # Store result in per-run cache for potential reuse.
        if cached is None and selected:
            cache.put(
                step.query, step.source_scope, plan.intent_route.retrieval_strategy,
                context_fingerprint, corpus_version, selected,
            )
        for chunk in selected:
            key = chunk_key(chunk)
            if key not in seen_chunks:
                seen_chunks.add(key)
                gathered.append(chunk)
        confidence = estimate_retrieval_confidence(request, gathered)
        attempt = RetrievalAttempt(
            step=step.step,
            query=step.query,
            tool_name=step.tool_name,
            source_scope=step.source_scope,
            hit_count=len(selected),
            source_types=sorted({chunk.source_type for chunk in selected}),
            selected_chunk_ids=[chunk_key(chunk) for chunk in selected],
            observation=summarize_observation(step.tool_name, selected) + observation_suffix,
            refined=step.step > 1,
            confidence=confidence,
            strategy=step.strategy,
            expanded_terms=step.expanded_terms,
            graph_edge_ids=[edge.edge_id for edge in (plan.graph_expansion.edges if plan.graph_expansion else [])],
        )
        attempts.append(attempt)
        trace.append(
            TraceEvent(
                event="rag.observe",
                node="retrieval_loop",
                status="completed",
                iteration=step.step,
                observation=attempt.observation,
                metadata={
                    "hit_count": attempt.hit_count,
                    "confidence": confidence,
                    "source_types": attempt.source_types,
                    "retrieval_mode": retrieval_mode,
                },
            )
        )
        if plan.intent_route.intent == "chat" and confidence >= plan.min_confidence:
            break
    trace.append(TraceEvent(event="graph.node.completed", node="retrieval_loop", status="completed"))
    return {
        "trace_events": trace,
        "retrieval_cache": cache,
        "retrieval_attempts": attempts,
        "agentic_context_chunks": gathered,
    }


def retrieve_context(state: GraphState) -> GraphState:
    """Retrieve context chunks from preloaded and agentic sources."""
    request = state["request"]
    preloaded = request.context_chunks
    agentic = state.get("agentic_context_chunks", [])
    combined: list[ContextChunk] = list(preloaded)
    seen: set[str] = set(chunk_key(c) for c in combined)
    for chunk in agentic:
        key = chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            combined.append(chunk)
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="retrieve_context", status="running"))
    trace.append(
        TraceEvent(
            event="retrieval.context_collected",
            node="retrieve_context",
            status="completed",
            metadata={"preloaded": len(preloaded), "agentic": len(agentic), "combined": len(combined)},
        )
    )
    trace.append(TraceEvent(event="graph.node.completed", node="retrieve_context", status="completed"))
    return {"trace_events": trace, "retrieved_context_chunks": combined}


def rerank_context(state: GraphState) -> GraphState:
    """Rerank retrieved context chunks using prepared candidates when available."""
    request = state["request"]
    chunks = state.get("retrieved_context_chunks", [])
    trace = state.get("trace_events", [])

    # Task 12: use prepared candidates for efficient reranking when the cache is warm.
    cache: RunRetrievalCache | None = state.get("retrieval_cache")
    if cache and cache.size > 0 and chunks:
        prepared = prepare_candidates(request.goal, chunks)
        output = rerank_prepared(prepared, top_k=8)
        trace.append(output.trace)
        trace.append(
            TraceEvent(
                event="retrieval.rerank_prepared",
                node="rerank_context",
                status="completed",
                metadata={"prepared_fingerprint": prepared.fingerprint, "candidate_count": len(prepared.chunks)},
            )
        )
        trace.append(TraceEvent(event="graph.node.completed", node="rerank_context", status="completed"))
        return {"trace_events": trace, "reranked_context_chunks": output.chunks}

    output = rerank_context_chunks(request.goal, chunks, limit=8)
    trace.append(output.trace)
    trace.append(TraceEvent(event="graph.node.completed", node="rerank_context", status="completed"))
    return {"trace_events": trace, "reranked_context_chunks": output.chunks}


def build_evidence_pack(state: GraphState) -> GraphState:
    """Build evidence pack from reranked context chunks."""
    request = state["request"]
    chunks = state.get("reranked_context_chunks", [])
    plan = state.get("retrieval_plan", RetrievalPlan())
    route = plan.intent_route
    graph = plan.graph_expansion
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="evidence_pack", status="running"))
    selected_ids: list[str] = []
    snippets: list[str] = []
    citations = citations_from_chunks(chunks)
    source_types = sorted({chunk.source_type for chunk in chunks})
    for chunk in chunks:
        key = chunk_key(chunk)
        if key not in selected_ids:
            selected_ids.append(key)
    for chunk in chunks[:5]:
        if chunk.snippet:
            snippets.append(chunk.snippet[:200])
    confidence = estimate_retrieval_confidence(request, chunks)
    coverage = len(set(source_types).intersection(route.required_source_types)) / max(len(route.required_source_types), 1)
    pack = EvidencePack(
        selected_chunk_ids=selected_ids,
        source_types=source_types,
        confidence=confidence,
        snippets=snippets,
        citations=citations,
        route_intent=route.intent,
        coverage=min(coverage, 1.0),
        graph_edges=graph.edges if graph else [],
    )
    trace.append(TraceEvent(event="graph.node.completed", node="evidence_pack", status="completed"))
    return {"trace_events": trace, "evidence_pack": pack}


def sufficiency_gate(state: GraphState) -> GraphState:
    """Evaluate context sufficiency and decide whether to proceed or refine."""
    request = state["request"]
    pack = state.get("evidence_pack", EvidencePack())
    result = evaluate_context_sufficiency(request, pack)
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="sufficiency_gate", status="running"))
    trace.append(
        TraceEvent(
            event="context.sufficiency",
            node="sufficiency_gate",
            status="completed" if result.sufficient else "insufficient",
            observation=result.reason,
            metadata={"sufficient": result.sufficient, "confidence": result.confidence, "missing_info": result.missing_info},
        )
    )
    trace.append(TraceEvent(event="graph.node.completed", node="sufficiency_gate", status="completed"))
    return {"trace_events": trace, "context_sufficiency": result}


def merge(state: GraphState) -> GraphState:
    """Merge role results into final output."""
    role_results = state.get("role_results", [])
    summary = first_non_empty(
        [item.summary for item in role_results if item.role == "summarizer"]
        + [item.summary for item in role_results]
    )
    action_items = unique_strings([item for role in role_results for item in role.action_items])
    risk_flags = unique_strings([item for role in role_results for item in role.risk_flags])
    citations = dedupe_citations([item for role in role_results for item in role.citations])
    if not summary:
        summary = "Python LangGraph meeting brief completed using the supplied meeting transcript context."
    if not action_items:
        action_items = synthesize_action_items(state["request"])
    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="merge", status="running"))
    trace.append(
        TraceEvent(
            event="graph.node.completed",
            node="merge",
            status="completed",
            metadata={"citations": len(citations), "risk_flags": len(risk_flags)},
        )
    )
    return {
        "trace_events": trace,
        "summary": summary,
        "action_items": action_items,
        "next_step": "Approve or reject the proposed write-back after checking citations.",
        "risk_flags": risk_flags,
        "citations": citations,
    }


def grounding_check(state: GraphState) -> GraphState:
    """Check grounding of summary against citations."""
    from ..grounding import check_grounding

    trace = state.get("trace_events", [])
    trace.append(TraceEvent(event="graph.node.started", node="grounding_check", status="running"))
    result = check_grounding(state.get("summary", ""), state.get("citations", []))
    trace.append(result.trace)
    trace.append(TraceEvent(event="graph.node.completed", node="grounding_check", status="completed"))
    return {
        "trace_events": trace,
        "grounding_check_result": {
            "grounded": result.grounded,
            "unsupported_claims": result.unsupported_claims,
        },
    }


def critic_check(state: GraphState) -> GraphState:
    """Run supervisor critic checks before the approval boundary."""
    trace = state.get("trace_events", [])
    sufficiency = state.get("context_sufficiency", ContextSufficiency())
    grounding = state.get("grounding_check_result", {})
    pack = state.get("evidence_pack", EvidencePack())
    plan = state.get("retrieval_plan", RetrievalPlan())
    attempts = state.get("retrieval_attempts", [])
    proposals = state.get("proposed_tool_calls", [])
    issues: list[str] = []
    grounding_passed = bool(grounding.get("grounded", True))
    if not sufficiency.sufficient:
        issues.append("insufficient_context_guarded")
    if not grounding_passed:
        issues.append("grounding_failed")
    if len(attempts) > plan.max_steps:
        issues.append("retrieval_budget_exceeded")
    write_safe = all(item.approval_required for item in proposals)
    if not write_safe:
        issues.append("unsafe_write_proposal")
    citation_coverage = pack.coverage
    if sufficiency.sufficient and pack.citations and citation_coverage <= 0:
        issues.append("citation_coverage_missing")
    result = CriticResult(
        passed=grounding_passed and write_safe and len(attempts) <= plan.max_steps,
        issues=issues,
        citation_coverage=citation_coverage,
        budget_respected=len(attempts) <= plan.max_steps,
        write_proposal_safe=write_safe,
        grounding_passed=grounding_passed,
        context_sufficient=sufficiency.sufficient,
    )
    trace.append(TraceEvent(event="graph.node.started", node="critic_check", status="running"))
    trace.append(
        TraceEvent(
            event="critic.check",
            node="critic_check",
            status="completed" if result.passed else "guarded",
            observation="; ".join(result.issues) if result.issues else "critic checks passed",
            metadata={
                "passed": result.passed,
                "issues": result.issues,
                "citation_coverage": result.citation_coverage,
                "budget_respected": result.budget_respected,
                "write_proposal_safe": result.write_proposal_safe,
                "grounding_passed": result.grounding_passed,
                "context_sufficient": result.context_sufficient,
            },
        )
    )
    trace.append(TraceEvent(event="graph.node.completed", node="critic_check", status="completed"))
    return {"trace_events": trace, "critic_result": result}



def _check_cancelled() -> None:
    """Raise ExecutionCancelled if the current execution deadline has been cancelled or expired."""
    deadline = get_current_deadline()
    if deadline is not None:
        deadline.raise_if_cancelled()
