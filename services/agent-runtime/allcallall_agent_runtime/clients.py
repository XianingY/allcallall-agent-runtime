"""Process-lifetime outbound HTTP clients for the Agent Runtime."""

from __future__ import annotations

import httpx

from .config import AgentRuntimeConfig
from .http_requests import build_http_client
from .providers.base import LLMProvider, create_provider
from .rag_runtime_client import RAGRuntimeClient
from .tool_layer import GoToolBridgeLayer


class RuntimeClients:
    """Bundle the Agent Runtime's outbound clients around one HTTP pool."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tool_bridge: GoToolBridgeLayer,
        rag_runtime: RAGRuntimeClient,
        owned_http: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.tool_bridge = tool_bridge
        self.rag_runtime = rag_runtime
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


def build_runtime_clients(
    config: AgentRuntimeConfig,
    http_client: httpx.Client | None = None,
) -> RuntimeClients:
    """Build the process-lifetime Agent Runtime client bundle."""
    owned: httpx.Client | None = None
    if http_client is None:
        http_client = build_http_client(config)
        owned = http_client

    provider: LLMProvider = create_provider(config=config, http_client=http_client)

    return RuntimeClients(
        provider=provider,
        tool_bridge=GoToolBridgeLayer(config=config, http_client=http_client),
        rag_runtime=RAGRuntimeClient(config=config, http_client=http_client),
        owned_http=owned,
    )
