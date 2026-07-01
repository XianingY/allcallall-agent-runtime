from __future__ import annotations

import math
import re

from shared.scoring import rules_score_rag, tokenize
from shared.utils import chunk_key

from .config import config
from .models import (
    AgenticRetrievalRequest,
    AgenticRetrievalResponse,
    ContextChunk,
    ContextSufficiency,
    EvidencePack,
    GraphExpansion,
    KnowledgeGraphEdge,
    GroundingCheckResponse,
    RetrievalAttempt,
    RetrievalRoute,
    RerankResponse,
)


def rerank(query: str, chunks: list[ContextChunk], top_k: int = 8) -> RerankResponse:
    tokens = tokenize(query, remove_stopwords=True)
    scored: list[tuple[float, ContextChunk]] = []
    for index, chunk in enumerate(chunks):
        score, reason = rules_score_rag(chunk, tokens, index)
        scored.append((score, chunk.model_copy(update={"rerank_score": score, "rerank_reason": reason})))
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [
        chunk.model_copy(update={"final_rank": index})
        for index, (_, chunk) in enumerate(scored[: max(1, top_k)], start=1)
    ]
    return RerankResponse(
        chunks=ranked,
        trace={
            "event": "retrieval.rerank",
            "provider": "rules",
            "candidate_count": len(chunks),
            "returned": len(ranked),
        },
    )


def agentic_retrieve(request: AgenticRetrievalRequest, chunks: list[ContextChunk]) -> AgenticRetrievalResponse:
    max_steps = max(1, min(request.max_steps, 3))
    route = route_query(request.query, request.source_types, chunks)
    graph = build_graph_expansion(request.query, chunks) if config.enable_graph_expansion else GraphExpansion()
    source_types = request.source_types or [
        item for item in route.required_source_types if item != "memory"
    ]
    attempts: list[RetrievalAttempt] = []
    gathered: list[ContextChunk] = []
    raw_candidates: list[ContextChunk] = []
    seen: set[str] = set()
    seen_raw: set[str] = set()
    trace: list[dict[str, object]] = [
        {
            "event": "rag.plan",
            "max_steps": max_steps,
            "query": request.query,
            "route": route.model_dump(),
            "graph_expansion": graph.model_dump(),
        }
    ]
    queries = build_queries(request.query, source_types, max_steps, route, graph)
    for step, query in enumerate(queries, start=1):
        scoped = filter_chunks(chunks, source_types)
        for chunk in scoped:
            key = chunk_key(chunk)
            if key not in seen_raw:
                seen_raw.add(key)
                raw_candidates.append(chunk)
        ranked = rerank(query, scoped, request.top_k).chunks
        for chunk in ranked:
            key = chunk_key(chunk)
            if key not in seen:
                seen.add(key)
                gathered.append(chunk)
        confidence = estimate_confidence(gathered, source_types, graph)
        attempts.append(
            RetrievalAttempt(
                step=step,
                query=query,
                source_types=sorted({chunk.source_type for chunk in ranked}),
                hit_count=len(ranked),
                selected_chunk_ids=[chunk_key(chunk) for chunk in ranked],
                confidence=confidence,
                observation=f"selected {len(ranked)} chunk(s), confidence={confidence:.2f}",
                refined=step > 1,
                strategy=route.retrieval_strategy,
                expanded_terms=graph.expanded_terms,
                graph_edge_ids=[edge.edge_id for edge in graph.edges],
            )
        )
        trace.append({"event": "rag.observe", "step": step, "hit_count": len(ranked), "confidence": confidence})
        if confidence >= request.min_confidence:
            break
    reranked_hits = rerank(request.query, raw_candidates or gathered, request.top_k).chunks
    pack = build_evidence_pack(gathered, source_types, route, graph)
    sufficiency = check_sufficiency(pack, source_types, request.min_confidence)
    selected_ids = set(pack.selected_chunk_ids)
    rejected_chunks = [chunk for chunk in raw_candidates if chunk_key(chunk) not in selected_ids]
    return AgenticRetrievalResponse(
        route=route,
        retrieval_route=route,
        graph_expansion=graph,
        attempts=attempts,
        raw_hits=raw_candidates[: max(1, request.top_k * max_steps)],
        reranked_hits=reranked_hits,
        rejected_chunks=rejected_chunks[: max(0, len(raw_candidates) - len(selected_ids))],
        evidence_pack=pack,
        context_sufficiency=sufficiency,
        trace=trace,
    )


def route_query(query: str, source_types: list[str], chunks: list[ContextChunk]) -> RetrievalRoute:
    text = " ".join([query] + [chunk.snippet for chunk in chunks[:6]]).lower()
    if source_types == ["none"]:
        return RetrievalRoute(
            intent="chat",
            target_workflow="react_general",
            confidence=0.8,
            rationale="caller requested no retrieval; answer should rely on supplied prompt only",
            required_source_types=[],
            retrieval_strategy="no_retrieval",
        )
    if "meeting_transcript" in source_types:
        return RetrievalRoute(
            intent="risk",
            target_workflow="risk_review",
            confidence=0.9,
            rationale="risk, approval, or transcript signal requires multi-hop evidence",
            required_source_types=["meeting_transcript", "knowledge", "memory"],
            retrieval_strategy="multi_hop",
        )
    if source_types == ["knowledge"] or contains_any(text, consult_keywords()):
        return RetrievalRoute(
            intent="consult",
            target_workflow="context_qa",
            confidence=0.84,
            rationale="question or policy signal requires knowledge-first retrieval",
            required_source_types=["knowledge", "meeting_transcript", "memory"],
            retrieval_strategy="graph_augmented",
        )
    if contains_any(text, risk_keywords()):
        return RetrievalRoute(
            intent="risk",
            target_workflow="risk_review",
            confidence=0.86,
            rationale="risk or approval signal requires multi-hop evidence",
            required_source_types=["meeting_transcript", "knowledge", "memory"],
            retrieval_strategy="multi_hop",
        )
    return RetrievalRoute(
        intent="chat",
        target_workflow="react_general",
        confidence=0.78,
        rationale="conversation task can use scoped chat and memory context",
        required_source_types=["message", "note", "followup", "memory"],
        retrieval_strategy="single_pass",
    )


def build_graph_expansion(query: str, chunks: list[ContextChunk]) -> GraphExpansion:
    query_terms = tokenize(query, remove_stopwords=True)[:8]
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
    expanded_terms = dedupe_terms(expanded, exclude=set(query_terms))[:12]
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
    text = f"{chunk.title} {chunk.source_title} {chunk.snippet}".strip()
    lowered = text.lower()
    relation = infer_relation(text)
    target = infer_target_phrase(text)
    if not relation or not target:
        return None
    has_overlap = not query_terms or any(term in lowered for term in query_terms)
    if not has_overlap and chunk.source_type not in {"knowledge", "meeting_transcript", "memory"}:
        return None
    source = chunk.title or chunk.source_title or chunk.source_type
    return KnowledgeGraphEdge(
        edge_id=f"kg-{ordinal}-{chunk.chunk_id or chunk.source_type + '-' + chunk.source_id}",
        source=source[:80],
        relation=relation,
        target=target[:120],
        evidence_chunk_id=chunk_key(chunk),
        confidence=0.72 if chunk.source_type == "knowledge" else 0.64,
    )


def grounding_check(answer: str, citations: list[ContextChunk]) -> GroundingCheckResponse:
    tokens = tokenize(answer, remove_stopwords=True)
    evidence = " ".join(chunk.snippet for chunk in citations).lower()
    if not tokens:
        return GroundingCheckResponse(grounded=False, unsupported_claims=["empty_answer"], coverage=0)
    covered = [token for token in tokens if token in evidence]
    coverage = len(covered) / max(len(tokens), 1)
    grounded = bool(citations) and coverage >= 0.2
    unsupported = [] if grounded else ["answer lacks enough overlap with supplied citations"]
    return GroundingCheckResponse(
        grounded=grounded,
        unsupported_claims=unsupported,
        coverage=coverage,
        trace={"event": "grounding.check", "citation_count": len(citations), "coverage": coverage},
    )


def filter_chunks(chunks: list[ContextChunk], source_types: list[str]) -> list[ContextChunk]:
    allowed = {item.strip() for item in source_types if item.strip()}
    if "none" in allowed:
        return []
    if not allowed:
        return chunks
    scoped = [chunk for chunk in chunks if chunk.source_type in allowed]
    return scoped or chunks


def build_queries(
    query: str,
    source_types: list[str],
    max_steps: int,
    route: RetrievalRoute | None = None,
    graph: GraphExpansion | None = None,
) -> list[str]:
    route = route or RetrievalRoute()
    graph = graph or GraphExpansion()
    expansion = " ".join(graph.expanded_terms[:6])
    queries = [query]
    if route.intent == "risk" or "meeting_transcript" in source_types:
        queries.append(f"{query} {expansion} meeting transcript decisions risks action items")
    if route.intent == "consult" or "knowledge" in source_types:
        queries.append(f"{query} {expansion} policy knowledge reference checklist")
    queries.append(f"{query} conversation notes followups")
    out: list[str] = []
    seen: set[str] = set()
    for item in queries:
        normalized = " ".join(item.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
        if len(out) >= max_steps:
            break
    return out


def build_evidence_pack(
    chunks: list[ContextChunk],
    required_source_types: list[str],
    route: RetrievalRoute | None = None,
    graph: GraphExpansion | None = None,
) -> EvidencePack:
    route = route or RetrievalRoute()
    graph = graph or GraphExpansion()
    ranked = rerank(" ".join(required_source_types + graph.expanded_terms), chunks, 8).chunks
    source_types = sorted({chunk.source_type for chunk in ranked})
    confidence = estimate_confidence(ranked, required_source_types, graph)
    coverage = len(set(source_types).intersection(required_source_types)) / max(len(required_source_types), 1)
    return EvidencePack(
        selected_chunk_ids=[chunk_key(chunk) for chunk in ranked],
        source_types=source_types,
        confidence=confidence,
        snippets=[chunk.snippet for chunk in ranked[:5]],
        citations=ranked[:5],
        route_intent=route.intent,
        coverage=min(coverage, 1.0),
        graph_edges=graph.edges,
    )


def check_sufficiency(pack: EvidencePack, required_source_types: list[str], threshold: float) -> ContextSufficiency:
    missing = [item for item in required_source_types if item not in pack.source_types]
    sufficient = not missing and pack.confidence >= threshold
    return ContextSufficiency(
        sufficient=sufficient,
        confidence=pack.confidence,
        reason="context is sufficient" if sufficient else "context is insufficient",
        missing_info=missing,
    )


def estimate_confidence(
    chunks: list[ContextChunk],
    required_source_types: list[str],
    graph: GraphExpansion | None = None,
) -> float:
    if not chunks:
        return 0.0
    graph = graph or GraphExpansion()
    sources = {chunk.source_type for chunk in chunks}
    confidence = min(0.45 + math.log1p(len(chunks)) * 0.2, 0.88)
    for source_type in required_source_types:
        if source_type in sources:
            confidence += 0.12
    if graph.enabled:
        confidence += min(len(graph.edges) * 0.03, 0.12)
    return min(confidence, 1.0)


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def risk_keywords() -> tuple[str, ...]:
    return (
        "risk",
        "approval",
        "blocker",
        "security",
        "legal",
        "deadline",
        "budget",
        "风险",
        "审批",
        "安全",
        "法务",
        "延期",
        "预算",
        "阻塞",
    )


def consult_keywords() -> tuple[str, ...]:
    return (
        "what",
        "how",
        "policy",
        "knowledge",
        "checklist",
        "requires",
        "为什么",
        "如何",
        "政策",
        "知识库",
        "清单",
        "材料",
    )


def infer_relation(text: str) -> str:
    relation_rules = [
        ("requires", ("requires", "require", "需要", "必须", "包含", "should")),
        ("blocks", ("blocker", "block", "风险", "延期", "delay")),
        ("owned_by", ("owner", "owns", "负责", "跟进", "action item")),
        ("approves", ("approval", "approve", "审批", "signoff")),
        ("mitigates", ("mitigation", "mitigate", "缓解", "回归", "rollback")),
    ]
    for relation, keywords in relation_rules:
        if contains_any(text, keywords):
            return relation
    return ""


def infer_target_phrase(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return ""
    for clause in re.split(r"[。；;.!?\n]", normalized):
        stripped = clause.strip()
        if stripped:
            return stripped
    return ""


def dedupe_terms(values: list[str], exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for term in tokenize(value, remove_stopwords=True):
            if term in exclude or term in seen:
                continue
            seen.add(term)
            out.append(term)
    return out
