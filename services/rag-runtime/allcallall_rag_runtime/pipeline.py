from __future__ import annotations

from .go_bridge import GoRetrievalBridge
from .models import ContextChunk, RetrievalQueryRequest
from .qdrant_adapter import QdrantAdapter, QdrantAdapterError


def select_retrieval_chunks(request: RetrievalQueryRequest) -> tuple[list[ContextChunk], str]:
    """Select authorized Go, optional Qdrant, or inline retrieval input."""
    bridge = GoRetrievalBridge()
    bridge_chunks = bridge.query(request) if bridge.configured() else []
    qdrant_chunks: list[ContextChunk] = []
    if not bridge_chunks:
        try:
            qdrant_chunks = QdrantAdapter().query(request)
        except QdrantAdapterError:
            qdrant_chunks = []
    if bridge_chunks:
        return bridge_chunks, "go_bridge"
    if qdrant_chunks:
        return qdrant_chunks, "qdrant"
    return request.chunks, "inline"
