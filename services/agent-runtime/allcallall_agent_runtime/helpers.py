"""Utility functions for the agent runtime graph."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .models import (
    Citation,
    ContextChunk,
    ContextSufficiency,
    EvidencePack,
    GraphExpansion,
    IntentRoute,
    KnowledgeGraphEdge,
    RetrievalPlanStep,
    WorkflowRequest,
)


# Tool name constants
READ_TOOL_CONTEXT_CHUNKS = "query_context_chunks"
READ_TOOL_KNOWLEDGE_CHUNKS = "query_knowledge_chunks"
READ_TOOL_MEETING_TRANSCRIPTS = "query_meeting_transcript_segments"
READ_TOOL_RECENT_FOLLOWUPS = "query_recent_followups"
READ_TOOL_RECENT_MEETINGS = "query_recent_meetings"
WRITE_CONVERSATION_MESSAGE = "write_conversation_message"
CREATE_FOLLOW_UP_TASK = "create_follow_up_task"
UPSERT_MEMORY = "upsert_agent_memory"

# Workflow constants
WORKFLOW_MEETING_BRIEF = "meeting_brief"
WORKFLOW_REACT_GENERAL = "react_general"
WORKFLOW_RISK_REVIEW = "risk_review"
WORKFLOW_FOLLOW_UP_PLANNER = "follow_up_planner"
WORKFLOW_CONTEXT_QA = "context_qa"

SUPPORTED_WORKFLOWS = {
    WORKFLOW_REACT_GENERAL,
    WORKFLOW_MEETING_BRIEF,
    WORKFLOW_RISK_REVIEW,
    WORKFLOW_FOLLOW_UP_PLANNER,
    WORKFLOW_CONTEXT_QA,
}

WORKFLOW_ALIASES = {
    "follow_up": WORKFLOW_FOLLOW_UP_PLANNER,
}


def chunk_key(chunk: ContextChunk) -> str:
    """Generate a unique key for a context chunk."""
    return chunk.chunk_id or f"{chunk.source_type}:{chunk.source_id}"


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Deduplicate citations by source_type and source_id."""
    out: list[Citation] = []
    seen: set[str] = set()
    for citation in citations:
        key = f"{citation.source_type}:{citation.source_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out


def unique_strings(values: list[str]) -> list[str]:
    """Deduplicate and normalize strings."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def first_non_empty(values: list[str]) -> str:
    """Return the first non-empty string from the list."""
    for value in values:
        if value.strip():
            return value
    return ""


def route_request_intent(request: WorkflowRequest) -> IntentRoute:
    """Route the request into chat, consult, or risk intent before retrieval.

    The preset is the strongest signal: ``risk_review`` and ``context_qa``
    short-circuit to their intent. Beyond the preset/goal, the router also
    inspects the full request surface (message bodies, notes, attachment text)
    via :func:`route_text` so that the *content* of a conversation drives
    routing instead of only the declared goal.
    """
    routing_text = route_text(request).lower()
    if request.preset == WORKFLOW_RISK_REVIEW:
        return IntentRoute(
            intent="risk",
            target_workflow=request.preset,
            confidence=0.9,
            rationale="risk workflow requires transcript, policy, and memory evidence",
            required_source_types=["meeting_transcript", "knowledge", "memory"],
            retrieval_strategy="multi_hop",
        )
    if request.preset == WORKFLOW_CONTEXT_QA or contains_any(routing_text, consult_keywords()):
        return IntentRoute(
            intent="consult",
            target_workflow=request.preset,
            confidence=0.84,
            rationale="question or policy language requires knowledge-grounded consultation",
            required_source_types=["knowledge", "meeting_transcript", "memory"],
            retrieval_strategy="graph_augmented",
        )
    if contains_any(routing_text, risk_keywords()):
        return IntentRoute(
            intent="risk",
            target_workflow=request.preset,
            confidence=0.86,
            rationale="risk language requires transcript, policy, and memory evidence",
            required_source_types=["meeting_transcript", "knowledge", "memory"],
            retrieval_strategy="multi_hop",
        )
    return IntentRoute(
        intent="chat",
        target_workflow=request.preset,
        confidence=0.78,
        rationale="conversation task can start from scoped chat, note, and memory context",
        required_source_types=["conversation", "message", "note", "followup", "memory"],
        retrieval_strategy="single_pass",
    )


def route_text(request: WorkflowRequest) -> str:
    """Build the user-authored text used for deterministic routing.

    Only the request surface the *user* explicitly authored is included: the
    preset, goal, conversation messages, and notes. Retrieved evidence
    (``context_chunks``, ``meeting_transcripts``) and attachment-derived text
    (OCR/caption/transcript) are deliberately excluded — routing must reflect
    the user's intent, not preloaded context or brittle attachment substrings.
    """
    parts = [request.preset, request.goal]
    parts.extend(message.body for message in request.messages)
    parts.extend(note.body for note in request.notes)
    return " ".join(part for part in parts if part)


def risk_keywords() -> tuple[str, ...]:
    return (
        "risk",
        "blocker",
        "approval",
        "security",
        "legal",
        "privacy",
        "deadline",
        "budget",
        "风险",
        "阻塞",
        "审批",
        "安全",
        "法务",
        "隐私",
        "延期",
        "预算",
        "升级",
    )


def consult_keywords() -> tuple[str, ...]:
    return (
        "what",
        "how",
        "why",
        "policy",
        "knowledge",
        "checklist",
        "require",
        "requires",
        "需要什么",
        "如何",
        "为什么",
        "政策",
        "知识库",
        "规范",
        "材料",
        "清单",
    )


def build_graph_expansion(query: str, chunks: list[ContextChunk]) -> GraphExpansion:
    """Infer lightweight evidence relationships for knowledge-graph enhanced RAG."""
    query_terms = tokenize_route_terms(query)[:8]
    edges: list[KnowledgeGraphEdge] = []
    expanded: list[str] = []
    for chunk in chunks:
        edge = graph_edge_from_chunk(chunk, query_terms, len(edges) + 1)
        if edge is None:
            continue
        edges.append(edge)
        expanded.extend([edge.source, edge.relation, edge.target])
        if len(edges) >= 8:
            break
    expanded_terms = [term for term in unique_strings(expanded) if term.lower() not in query_terms][:12]
    return GraphExpansion(
        enabled=bool(edges),
        query_terms=query_terms,
        expanded_terms=expanded_terms,
        edges=edges,
    )


def graph_edge_from_chunk(
    chunk: ContextChunk,
    query_terms: list[str],
    ordinal: int,
) -> KnowledgeGraphEdge | None:
    """Create one deterministic relationship from a chunk when it overlaps the query."""
    text = f"{chunk.title} {chunk.source_title} {chunk.snippet}"
    lowered = text.lower()
    relation = infer_relation(text)
    if not relation:
        return None
    has_overlap = not query_terms or any(term.lower() in lowered for term in query_terms)
    if not has_overlap and chunk.source_type not in {"knowledge", "meeting_transcript", "memory"}:
        return None
    source = first_non_empty([chunk.title, chunk.source_title, chunk.source_type])
    target = infer_target_phrase(text)
    if not target:
        return None
    return KnowledgeGraphEdge(
        edge_id=f"kg-{ordinal}-{chunk.chunk_id or chunk.source_type + '-' + chunk.source_id}",
        source=source[:80],
        relation=relation,
        target=target[:120],
        evidence_chunk_id=chunk_key(chunk),
        confidence=0.72 if chunk.source_type == "knowledge" else 0.64,
    )


def infer_relation(text: str) -> str:
    lowered = text.lower()
    relation_rules = [
        ("requires", ("requires", "require", "需要", "必须", "包含", "should")),
        ("blocks", ("blocker", "block", "阻塞", "风险", "延期", "delay")),
        ("owned_by", ("owner", "负责", "跟进", "action item", "owns")),
        ("approves", ("approval", "approve", "审批", "signoff")),
        ("mitigates", ("mitigation", "mitigate", "缓解", "回归", "rollback")),
    ]
    for relation, keywords in relation_rules:
        if contains_any(lowered, keywords):
            return relation
    return ""


def infer_target_phrase(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return ""
    clauses = re.split(r"[。；;.!?\n]", normalized)
    return first_non_empty([clause.strip() for clause in clauses])


def tokenize_route_terms(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()):
        token = token.strip()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def citations_from_chunks(chunks: list[ContextChunk]) -> list[Citation]:
    """Extract citations from context chunks."""
    return dedupe_citations(
        [
            Citation(
                chunk_id=chunk.chunk_id,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                source_title=chunk.source_title or chunk.title,
                title=chunk.title or chunk.source_title or f"{chunk.source_type} #{chunk.source_id}",
                snippet=chunk.snippet,
                score=chunk.score,
                retrieval_mode=chunk.retrieval_mode,
                rerank_score=chunk.rerank_score,
                rerank_reason=chunk.rerank_reason,
                final_rank=chunk.final_rank,
                recording_session_id=chunk.recording_session_id,
                recording_file_id=chunk.recording_file_id,
                transcript_segment_id=chunk.transcript_segment_id,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
            for chunk in chunks
            if chunk.source_type and chunk.source_id and chunk.snippet
        ]
    )


def top_snippets(chunks: list[ContextChunk], limit: int) -> list[str]:
    """Extract top snippets from context chunks."""
    return unique_strings([chunk.snippet for chunk in chunks if chunk.snippet])[:limit]


def summarize_observation(tool_name: str, chunks: list[ContextChunk]) -> str:
    """Summarize the observation from tool execution."""
    if tool_name == READ_TOOL_RECENT_MEETINGS:
        return "recent meeting metadata inspected"
    if not chunks:
        return "0 chunks"
    sources = ", ".join(unique_strings([chunk.source_type for chunk in chunks]))
    return f"{len(chunks)} chunks from {sources}: {chunks[0].snippet[:120]}"


def normalize_workflow_preset(raw: str) -> str:
    """Normalize a workflow preset name."""
    normalized = raw.strip() or WORKFLOW_MEETING_BRIEF
    return WORKFLOW_ALIASES.get(normalized, normalized)


def tool_allowed(request: WorkflowRequest, tool_name: str) -> bool:
    """Check if a tool is allowed by the request's tool policy."""
    allowed = set(request.tool_policy.read_tools or [])
    return not allowed or tool_name in allowed


def runtime_subject_id(request: WorkflowRequest) -> str:
    """Generate a subject ID for runtime operations."""
    if request.agent_run_id:
        return f"agent:{request.agent_run_id}"
    return f"workflow:{request.workflow_run_id}"


def request_with_runtime_context(state: Mapping[str, Any]) -> WorkflowRequest:
    """Get request with runtime context (reranked or retrieved chunks)."""
    request = WorkflowRequest.model_validate(state["request"])
    chunks = state.get("reranked_context_chunks") or state.get("retrieved_context_chunks")
    if chunks is None:
        return request
    return request.model_copy(update={"context_chunks": chunks})


def estimate_retrieval_confidence(request: WorkflowRequest, chunks: list[ContextChunk]) -> float:
    """Estimate confidence of retrieval based on chunks and workflow preset."""
    if not chunks:
        return 0.0
    source_types = {chunk.source_type for chunk in chunks}
    confidence = min(0.40 + len(chunks) * 0.08, 0.75)
    if "meeting_transcript" in source_types:
        confidence += 0.18
    if "knowledge" in source_types:
        confidence += 0.12
    if request.preset == WORKFLOW_CONTEXT_QA and "knowledge" not in source_types:
        confidence -= 0.25
    if request.preset == WORKFLOW_MEETING_BRIEF and "meeting_transcript" not in source_types:
        confidence -= 0.2
    if request.preset == WORKFLOW_RISK_REVIEW and source_types.intersection({"conversation", "message", "note"}):
        confidence += 0.12
    if any(chunk.source_type == "memory" for chunk in chunks):
        confidence += 0.05
    return max(0.0, min(confidence, 1.0))


def evaluate_context_sufficiency(request: WorkflowRequest, pack: EvidencePack) -> ContextSufficiency:
    """Evaluate whether context is sufficient for grounded synthesis."""
    missing: list[str] = []
    if not pack.citations:
        missing.append("retrieved evidence")
    if request.preset == WORKFLOW_MEETING_BRIEF and "meeting_transcript" not in pack.source_types:
        missing.append("meeting transcript citation")
    if request.preset == WORKFLOW_CONTEXT_QA and not pack.source_types:
        missing.append("knowledge or conversation evidence")
    risk_sources = {"meeting_transcript", "conversation", "message", "note"}
    if pack.route_intent == "risk" and not risk_sources.intersection(pack.source_types):
        missing.append("risk evidence")
    if pack.route_intent == "consult" and "knowledge" not in pack.source_types:
        missing.append("knowledge citation")
    threshold = 0.55 if pack.route_intent in {"risk", "consult"} else 0.45
    sufficient = not missing and pack.confidence >= threshold
    reason = "context is sufficient for grounded synthesis" if sufficient else "context is insufficient for grounded synthesis"
    return ContextSufficiency(
        sufficient=sufficient,
        confidence=pack.confidence,
        reason=reason,
        missing_info=missing,
    )


def local_agentic_retrieval(chunks: list[ContextChunk], step: RetrievalPlanStep) -> list[ContextChunk]:
    """Perform local agentic retrieval from preloaded chunks."""
    from .retrieval import rerank_context_chunks

    scoped = chunks
    if step.source_scope == "knowledge":
        scoped = [chunk for chunk in chunks if chunk.source_type == "knowledge"]
    elif step.source_scope == "meeting_transcript":
        scoped = [chunk for chunk in chunks if chunk.source_type == "meeting_transcript"]
    elif step.source_scope in {"conversation", "message", "note", "followup", "memory", "contact_profile"}:
        scoped = [chunk for chunk in chunks if chunk.source_type in {step.source_scope, "conversation"}]
    if not scoped and step.source_scope != "all":
        scoped = chunks
    output = rerank_context_chunks(step.query, scoped, limit=6)
    return output.chunks


def select_chunks(chunks: list[ContextChunk], role: str, iteration: int) -> list[ContextChunk]:
    """Select chunks for the ReAct loop based on role and iteration."""
    if not chunks:
        return []
    preferred = []
    if role == "risk_analyst":
        keywords = ("risk", "approval", "blocker", "deadline", "budget", "security", "legal")
        preferred = [chunk for chunk in chunks if contains_any(chunk.snippet, keywords)]
    if not preferred:
        preferred = [chunk for chunk in chunks if chunk.source_type == "meeting_transcript"]
    if not preferred:
        preferred = chunks
    limit = 5 if iteration == 1 else 4
    return preferred[:limit]
