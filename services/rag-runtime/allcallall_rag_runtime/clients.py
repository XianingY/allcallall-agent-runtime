"""Process-lifetime outbound HTTP clients for the RAG Runtime."""

from __future__ import annotations

import httpx

from .config import RAGRuntimeConfig
from .go_bridge import GoRetrievalBridge
from .qdrant_adapter import QdrantAdapter


def _http_client(config: RAGRuntimeConfig) -> httpx.Client:
    """Create a bounded, process-owned HTTP client."""
    return httpx.Client(
        limits=httpx.Limits(
            max_connections=config.http_max_connections,
            max_keepalive_connections=config.http_max_keepalive_connections,
            keepalive_expiry=config.http_keepalive_expiry_sec,
        ),
        timeout=httpx.Timeout(
            connect=config.http_connect_timeout_sec,
            read=config.http_read_timeout_sec,
            write=config.http_write_timeout_sec,
            pool=config.http_pool_timeout_sec,
        ),
    )


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
        http_client = _http_client(config)
        owned = http_client

    return RAGClients(
        go_bridge=GoRetrievalBridge(config=config, http_client=http_client),
        qdrant=QdrantAdapter(config=config, http_client=http_client),
        owned_http=owned,
    )
