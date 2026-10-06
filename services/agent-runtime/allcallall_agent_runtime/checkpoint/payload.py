"""Checkpoint payload projection: slim checkpoint state before serialization.

Removes request-scoped clients and per-run caches while retaining durable
graph state.  Chunk and citation lists remain in their serializer-native
representation because existing graph nodes access them as model objects after
a resume.  Trace payloads are compacted where nodes only append to the list.

The projection is applied immediately before serialization so the durable
checkpoint stays within the 16 MiB transaction limit while preserving all
state needed to resume a paused workflow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


# Keys that are always removed because they hold non-serializable runtime objects
# or per-run caches that must not be persisted.
_EXCLUDED_KEYS: frozenset[str] = frozenset({
    "provider",       # LLMProvider — re-injected by the harness on resume
    "tool_bridge",    # GoToolBridge — re-injected by the harness on resume
    "rag_runtime",    # RAGRuntimeClient — re-injected by the harness on resume
    "retrieval_cache",  # RunRetrievalCache — per-run, not serializable
})

def project_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project graph state into a compact, serializable form for checkpointing.

    - Removes request-scoped clients and per-run caches.
    - Preserves agentic, retrieved, reranked, and citation model lists
      unchanged so resumed nodes continue to receive model objects.
    - Compacts trace events to event counts and key metadata.
    - Retains all state needed for resume: execution ID, checkpoint version,
      pending approvals, selected evidence IDs, loop position, and compact
      role outputs.

    The returned dict is safe to serialize and sufficient for resuming a
    paused workflow.  The harness re-injects ``provider``, ``tool_bridge``,
    and ``rag_runtime`` on every invoke, so dropping them from the durable
    checkpoint is safe.
    """
    projected: dict[str, Any] = {}
    for key, value in state.items():
        if key in _EXCLUDED_KEYS:
            continue
        if key == "trace_events":
            projected[key] = _compact_trace_events(value)
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
