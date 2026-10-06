from __future__ import annotations

import logging

from .models import ContextChunk, RetrievalQueryRequest
from .clients import RAGClients, build_rag_clients
from .metrics import metrics
from .qdrant_adapter import QdrantAdapterError
from .config import config as default_config

logger = logging.getLogger(__name__)



def select_retrieval_chunks(
    request: RetrievalQueryRequest,
    clients: RAGClients | None = None,
) -> tuple[list[ContextChunk], str]:
    """Select authorized Go, optional Qdrant, or inline retrieval input."""
    owned_clients: RAGClients
    if clients is not None:
        owned_clients = clients
    else:
        owned_clients = build_rag_clients(default_config)
    try:
        bridge = owned_clients.go_bridge
        bridge_chunks = bridge.query(request) if bridge.configured() else []
        qdrant_chunks: list[ContextChunk] = []
        if not bridge_chunks:
            try:
                qdrant_chunks = owned_clients.qdrant.query(request)
            except QdrantAdapterError as exc:
                metrics.inc("rag_runtime_qdrant_fallback_total")
                logger.warning(
                    "rag_runtime_qdrant_fallback",
                    extra={"error_type": exc.error_type},
                )
                qdrant_chunks = []
    finally:
        if clients is None:
            owned_clients.close()
    if bridge_chunks:
        return bridge_chunks, "go_bridge"
    if qdrant_chunks:
        return qdrant_chunks, "qdrant"
    return request.chunks, "inline"
