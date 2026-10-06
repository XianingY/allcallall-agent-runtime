"""Process-lifetime outbound HTTP clients for the RAG Runtime."""

from __future__ import annotations

import httpx

from .config import RAGRuntimeConfig
from .go_bridge import GoRetrievalBridge
from .http_requests import build_http_client
from .qdrant_adapter import QdrantAdapter


class RAGClients:
    """Bundle the RAG Runtime's outbound clients around one HTTP pool."""

    def __init__(
        self,
        *,
        go_bridge: GoRetrievalBridge,
        qdrant: QdrantAdapter,
        owned_http: httpx.Client | None = None,
    ) -> None:
        self.go_bridge = go_bridge
        self.qdrant = qdrant
        self._owned_http = owned_http
        self._closed = False

    @property
    def owned_http(self) -> httpx.Client | None:
        """Return the client owned by this bundle, if any."""
        return self._owned_http

    @property
    def closed(self) -> bool:
        """Whether close has already run."""
        return self._closed

    def close(self) -> None:
        """Close the owned HTTP pool once; injected clients stay caller-owned."""
        if self._closed:
            return
        self._closed = True
        if self._owned_http is not None:
            self._owned_http.close()
            self._owned_http = None


def build_rag_clients(
    config: RAGRuntimeConfig,
    http_client: httpx.Client | None = None,
) -> RAGClients:
    """Build the process-lifetime RAG Runtime client bundle."""
    owned: httpx.Client | None = None
    if http_client is None:
        http_client = build_http_client(config)
        owned = http_client

    return RAGClients(
        go_bridge=GoRetrievalBridge(config=config, http_client=http_client),
        qdrant=QdrantAdapter(config=config, http_client=http_client),
        owned_http=owned,
    )
