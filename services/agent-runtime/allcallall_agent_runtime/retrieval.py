"""Retrieval helpers: reranking, per-run cache, and candidate preparation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .models import ContextChunk, TraceEvent


CacheKey = tuple[str, str, str, str, str]
"""Cache key: (normalized_query, source_scope, retrieval_policy, context_fingerprint, corpus_version)."""


@dataclass(frozen=True)
class RerankOutput:
    chunks: list[ContextChunk]
    trace: TraceEvent


@dataclass(frozen=True)
class PreparedCandidates:
    """Normalized, filtered, deduplicated, and tokenized candidates ready for reranking.

    Computed once per unique (query, source_types, context_fingerprint, corpus_version)
    and reused when those inputs are unchanged, avoiding redundant reranking.
    """

    query: str
    source_types: tuple[str, ...]
    chunks: tuple[ContextChunk, ...]
    tokens: tuple[str, ...]
    fingerprint: str  # SHA-256 prefix of chunk keys for change detection


@dataclass
class RunRetrievalCache:
    """Per-run retrieval cache keyed by normalized query, source scope, retrieval
    policy, context fingerprint, and corpus version.

    Bounded to one execution, ``max_entries`` entries (default 16), and the
    request's chunk budget.  Not serialized into checkpoints — a resumed run
    starts with a fresh cache.
    """

    max_entries: int = 16
    _cache: dict[CacheKey, list[ContextChunk]] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    def get(
        self,
        query: str,
        source_scope: str,
        policy: str,
        context_fingerprint: str,
        corpus_version: str,
    ) -> list[ContextChunk] | None:
        key = _make_cache_key(query, source_scope, policy, context_fingerprint, corpus_version)
        entry = self._cache.get(key)
        if entry is not None:
            self._hits += 1
            return entry
        self._misses += 1
        return None

    def put(
        self,
        query: str,
        source_scope: str,
        policy: str,
        context_fingerprint: str,
        corpus_version: str,
        chunks: list[ContextChunk],
    ) -> None:
        key = _make_cache_key(query, source_scope, policy, context_fingerprint, corpus_version)
        if key not in self._cache and len(self._cache) >= self.max_entries:
            # Evict the oldest entry (first inserted).
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = chunks

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        return len(self._cache)


def _make_cache_key(
    query: str,
    source_scope: str,
    policy: str,
    context_fingerprint: str,
    corpus_version: str,
) -> CacheKey:
    normalized_query = " ".join(query.lower().split())
    return (normalized_query, source_scope, policy, context_fingerprint, corpus_version)


def compute_context_fingerprint(chunks: list[ContextChunk]) -> str:
    """Compute a stable fingerprint for a list of context chunks.

    Uses sorted chunk keys so the fingerprint is order-independent.
    Returns a 16-character hex prefix of the SHA-256 digest.
    """
    if not chunks:
        return ""
    keys = sorted(_chunk_key(c) for c in chunks)
    digest = hashlib.sha256("|".join(keys).encode()).hexdigest()
    return digest[:16]


def _chunk_key(chunk: ContextChunk) -> str:
    return chunk.chunk_id or f"{chunk.source_type}:{chunk.source_id}"


def prepare_candidates(
    query: str,
    chunks: list[ContextChunk],
    source_types: list[str] | None = None,
) -> PreparedCandidates:
    """Normalize, filter, deduplicate, and tokenize chunks into PreparedCandidates.

    Filtering by ``source_types`` narrows the candidate pool before expensive
    reranking.  Deduplication is by chunk key.  The fingerprint is a SHA-256
    prefix of the sorted chunk keys so change detection is order-independent.
    """
    # Filter by source types
    if source_types and source_types != ["all"]:
        filtered = [c for c in chunks if c.source_type in source_types]
    else:
        filtered = list(chunks)

    # Deduplicate by chunk key
    seen: set[str] = set()
    deduped: list[ContextChunk] = []
    for chunk in filtered:
        key = _chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    # Tokenize query
    tokens = tuple(tokenize(query))

    # Compute fingerprint
    fingerprint = compute_context_fingerprint(deduped)

    return PreparedCandidates(
        query=query,
        source_types=tuple(sorted(source_types)) if source_types else (),
        chunks=tuple(deduped),
        tokens=tokens,
        fingerprint=fingerprint,
    )


def rerank_prepared(prepared: PreparedCandidates, top_k: int = 8) -> RerankOutput:
    """Rerank prepared candidates.  Applies cheap top-N reduction before
    expensive reranking when the candidate pool exceeds 2x top_k."""
    if not prepared.chunks:
        return RerankOutput(
            chunks=[],
            trace=TraceEvent(
                event="retrieval.rerank",
                node="rerank_context",
                status="completed",
                metadata={"provider": "rules", "candidate_count": 0, "prepared_reuse": True},
            ),
        )
    # Cheap top-N pre-filter: only rerank top 2*top_k candidates by original score.
    candidates = list(prepared.chunks)
    if len(candidates) > top_k * 2:
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: top_k * 2]
    return rerank_context_chunks(prepared.query, candidates, limit=top_k)


def retrieve_context_chunks(chunks: list[ContextChunk]) -> list[ContextChunk]:
    return list(chunks)


def rerank_context_chunks(query: str, chunks: list[ContextChunk], limit: int = 8) -> RerankOutput:
    if not chunks:
        return RerankOutput(
            chunks=[],
            trace=TraceEvent(
                event="retrieval.rerank",
                node="rerank_context",
                status="completed",
                metadata={"provider": "rules", "candidate_count": 0},
            ),
        )
    tokens = tokenize(query)
    scored: list[tuple[float, ContextChunk, str]] = []
    for index, chunk in enumerate(chunks):
        score, reason = rules_score(chunk, tokens, index)
        updated = chunk.model_copy(
            update={
                "rerank_score": score,
                "rerank_reason": reason,
                "final_rank": 0,
            }
        )
        scored.append((score, updated, reason))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[ContextChunk] = []
    for index, (_, chunk, _) in enumerate(scored[:limit], start=1):
        out.append(chunk.model_copy(update={"final_rank": index}))
    return RerankOutput(
        chunks=out,
        trace=TraceEvent(
            event="retrieval.rerank",
            node="rerank_context",
            status="completed",
            metadata={
                "provider": "rules",
                "candidate_count": len(chunks),
                "returned": len(out),
                "top": [
                    {
                        "source_type": chunk.source_type,
                        "source_id": chunk.source_id,
                        "rerank_score": chunk.rerank_score,
                        "final_rank": chunk.final_rank,
                    }
                    for chunk in out[:5]
                ],
            },
        ),
    )


def rules_score(chunk: ContextChunk, tokens: list[str], original_index: int) -> tuple[float, str]:
    text = f"{chunk.title} {chunk.source_title} {chunk.snippet}".lower()
    title = f"{chunk.title} {chunk.source_title}".lower()
    overlap = sum(1 for token in tokens if token and token in text)
    title_overlap = sum(1 for token in tokens if token and token in title)
    source_boost = {
        "meeting_transcript": 5.0,
        "knowledge": 4.0,
        "transcript": 3.0,
        "followup": 2.5,
        "memory": 2.0,
        "note": 1.5,
        "message": 1.0,
    }.get(chunk.source_type, 0.0)
    base = max(chunk.score, 0) / 100.0
    score = overlap * 10.0 + title_overlap * 8.0 + source_boost + base - original_index * 0.01
    return score, f"rules keyword_overlap={overlap} title_overlap={title_overlap} source={chunk.source_type}"


def tokenize(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()):
        token = token.strip()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out
