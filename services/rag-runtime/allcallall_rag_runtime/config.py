from __future__ import annotations

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

    model_config = {"env_prefix": "PY_RAG_"}


config = RAGRuntimeConfig()
