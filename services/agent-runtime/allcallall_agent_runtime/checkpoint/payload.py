"""Checkpoint payload projection: slim checkpoint state before serialization.

Removes request-scoped clients, cancellation objects, duplicated context,
complete provider bodies, and complete trace payloads.  Large evidence bodies
are replaced with references and hashes while retaining resume state.

The projection is applied immediately before serialization so the durable
checkpoint stays within the 16 MiB transaction limit while preserving all
state needed to resume a paused workflow.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


# Keys that are always removed because they hold non-serializable runtime objects
# or per-run caches that must not be persisted.
_EXCLUDED_KEYS: frozenset[str] = frozenset({
    "provider",       # LLMProvider — re-injected by the harness on resume
    "tool_bridge",    # GoToolBridge — re-injected by the harness on resume
    "retrieval_cache",  # RunRetrievalCache — per-run, not serializable
})

# Keys whose values are large chunk lists that should be compacted to references.
_CHUNK_LIST_KEYS: frozenset[str] = frozenset({
    "agentic_context_chunks",
    "retrieved_context_chunks",
    "reranked_context_chunks",
})

# Maximum snippet length retained in compacted citations (characters).
_MAX_COMPACT_SNIPPET = 120


def project_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project graph state into a compact, serializable form for checkpointing.

    - Removes request-scoped clients and per-run caches.
    - Compacts large chunk lists into reference lists (chunk_id, source_type,
      source_id, hash) so the full snippet text is not duplicated.
    - Compacts trace events to event counts and key metadata.
    - Compacts citations to IDs and scores, dropping full snippet text.
    - Retains all state needed for resume: execution ID, checkpoint version,
      pending approvals, selected evidence IDs, loop position, and compact
      role outputs.

    The returned dict is safe to serialize and sufficient for resuming a
    paused workflow.  The harness re-injects ``provider`` and ``tool_bridge``
    on every invoke, so dropping them from the durable checkpoint is safe.
    """
    projected: dict[str, Any] = {}
    for key, value in state.items():
        if key in _EXCLUDED_KEYS:
            continue
        if key in _CHUNK_LIST_KEYS:
            projected[key] = _compact_chunk_list(value)
        elif key == "trace_events":
            projected[key] = _compact_trace_events(value)
        elif key == "citations":
            projected[key] = _compact_citations(value)
        else:
            # Keep everything else as-is (role results, evidence pack, etc.)
            projected[key] = value
    return projected


def serialized_checkpoint_size(payload: Mapping[str, Any]) -> int:
    """Return the byte size of a checkpoint payload when serialized as JSON.

    Used to emit original vs. projected byte histograms so operators can
    monitor checkpoint size and detect regressions.
    """
    try:
        return len(json.dumps(payload, default=_json_default, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        # Fallback: repr-based estimate for non-JSON-serializable values.
        return len(repr(payload).encode("utf-8"))


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types (e.g. Pydantic models, dataclasses)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _compact_chunk_list(chunks: Any) -> list[dict[str, str]]:
    """Replace a list of ContextChunk objects with compact references.

    Each reference retains chunk_id, source_type, source_id, and a SHA-256
    prefix of the snippet for integrity verification, but drops the full
    snippet text which is the primary source of checkpoint bloat.
    """
    if not isinstance(chunks, (list, tuple)):
        return []
    refs: list[dict[str, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            # Pydantic model or similar — extract fields.
            chunk_id = getattr(chunk, "chunk_id", "") or ""
            source_type = getattr(chunk, "source_type", "") or ""
            source_id = getattr(chunk, "source_id", "") or ""
            snippet = getattr(chunk, "snippet", "") or ""
        else:
            chunk_id = chunk.get("chunk_id", "") or ""
            source_type = chunk.get("source_type", "") or ""
            source_id = chunk.get("source_id", "") or ""
            snippet = chunk.get("snippet", "") or ""
        snippet_hash = hashlib.sha256(snippet.encode()).hexdigest()[:16] if snippet else ""
        refs.append({
            "chunk_id": chunk_id,
            "source_type": source_type,
            "source_id": source_id,
            "snippet_hash": snippet_hash,
        })
    return refs


def _compact_trace_events(events: Any) -> list[dict[str, Any]]:
    """Compact trace events: keep event name, node, status, and a metadata summary.

    Full tool_input and observation strings are dropped to reduce size.
    The compact form retains enough information for audit and debugging
    without the overhead of complete payloads.
    """
    if not isinstance(events, (list, tuple)):
        return []
    compacted: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            compacted.append({
                "event": event.get("event", ""),
                "node": event.get("node", ""),
                "status": event.get("status", ""),
                "iteration": event.get("iteration"),
                "role": event.get("role", ""),
                "tool_name": event.get("tool_name", ""),
            })
        else:
            # Pydantic model
            compacted.append({
                "event": getattr(event, "event", ""),
                "node": getattr(event, "node", ""),
                "status": getattr(event, "status", ""),
                "iteration": getattr(event, "iteration", None),
                "role": getattr(event, "role", ""),
                "tool_name": getattr(event, "tool_name", ""),
            })
    return compacted


def _compact_citations(citations: Any) -> list[dict[str, Any]]:
    """Compact citations: retain IDs and scores, truncate snippet text.

    The full snippet is not needed for resume; the chunk_id and source_id
    are sufficient to reconstruct the reference.  A truncated snippet is
    kept for human readability in checkpoint inspection.
    """
    if not isinstance(citations, (list, tuple)):
        return []
    compacted: list[dict[str, Any]] = []
    for cite in citations:
        if isinstance(cite, dict):
            snippet = cite.get("snippet", "") or ""
            compacted.append({
                "chunk_id": cite.get("chunk_id", ""),
                "source_type": cite.get("source_type", ""),
                "source_id": cite.get("source_id", ""),
                "score": cite.get("score", 0),
                "rerank_score": cite.get("rerank_score", 0),
                "final_rank": cite.get("final_rank", 0),
                "snippet": snippet[:_MAX_COMPACT_SNIPPET],
            })
        else:
            snippet = getattr(cite, "snippet", "") or ""
            compacted.append({
                "chunk_id": getattr(cite, "chunk_id", ""),
                "source_type": getattr(cite, "source_type", ""),
                "source_id": getattr(cite, "source_id", ""),
                "score": getattr(cite, "score", 0),
                "rerank_score": getattr(cite, "rerank_score", 0),
                "final_rank": getattr(cite, "final_rank", 0),
                "snippet": snippet[:_MAX_COMPACT_SNIPPET],
            })
    return compacted
