"""Process-lifetime outbound HTTP clients for the Agent Runtime."""

from __future__ import annotations

import httpx

from .config import AgentRuntimeConfig
from .providers.base import LLMProvider, RulesProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .rag_runtime_client import RAGRuntimeClient
from .tool_layer import GoToolBridgeLayer


def _http_client(config: AgentRuntimeConfig) -> httpx.Client:
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
        http_client = _http_client(config)
        owned = http_client

    if config.provider.lower() == "openai_compatible":
        provider: LLMProvider = OpenAICompatibleProvider(config=config, http_client=http_client)
    else:
        provider = RulesProvider()

    return RuntimeClients(
        provider=provider,
        tool_bridge=GoToolBridgeLayer(config=config, http_client=http_client),
        rag_runtime=RAGRuntimeClient(config=config, http_client=http_client),
        owned_http=owned,
    )
