from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings


class RAGRuntimeConfig(BaseSettings):
    """Centralized configuration for the RAG Runtime.

    All settings are loaded from environment variables with the ``PY_RAG_`` prefix.
    """

    # Tool bridge settings
    tool_bridge_base_url: str = ""
    tool_bridge_token: str = ""
    tool_bridge_timeout_sec: float = 10.0

    # Rerank settings
    rerank_provider: str = "rules"
    top_k: int = 8
    max_steps: int = 3
    min_confidence: float = 0.6
    enable_graph_expansion: bool = True
    enable_llamaindex_baseline: bool = False

    # Optional vector store adapter. Production AllCallAll still authorizes retrieval via Go.
    vector_store: str = "none"
    qdrant_url: str = ""
    qdrant_collection: str = "allcallall_context_chunks"
    qdrant_api_key: str = ""
    qdrant_timeout_sec: float = 5.0

    # Outbound HTTP connection pool (process-lifetime client bundle)
    http_max_connections: int = 20
    http_max_keepalive_connections: int = 10
    http_keepalive_expiry_sec: float = 30.0
    http_connect_timeout_sec: float = 5.0
    http_read_timeout_sec: float = 30.0
    http_write_timeout_sec: float = 10.0
    http_pool_timeout_sec: float = 10.0

    @model_validator(mode="after")
    def _validate_http_pool(self) -> Self:
        http_fields = (
            self.http_max_connections,
            self.http_max_keepalive_connections,
            self.http_keepalive_expiry_sec,
            self.http_connect_timeout_sec,
            self.http_read_timeout_sec,
            self.http_write_timeout_sec,
            self.http_pool_timeout_sec,
        )
        if any(value <= 0 for value in http_fields):
            raise ValueError("all outbound HTTP pool and timeout settings must be positive")
        if self.http_max_keepalive_connections > self.http_max_connections:
            raise ValueError("http_max_keepalive_connections cannot exceed http_max_connections")
        return self

    model_config = {"env_prefix": "PY_RAG_"}


config = RAGRuntimeConfig()
