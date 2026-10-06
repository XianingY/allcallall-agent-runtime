from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import RAGRuntimeConfig, config as default_config
from .http_requests import build_http_client, post_json_without_cookies
from .metrics import metrics
from .models import ContextChunk, RetrievalQueryRequest

logger = logging.getLogger(__name__)


class GoRetrievalBridge:
    def __init__(
        self,
        *,
        config: RAGRuntimeConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        settings = config or default_config
        self._settings = settings
        self.base_url = settings.tool_bridge_base_url.strip().rstrip("/")
        self.token = settings.tool_bridge_token.strip()
        self.timeout_sec = int(settings.tool_bridge_timeout_sec)
        self._http = http_client

    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def query(self, request: RetrievalQueryRequest) -> list[ContextChunk]:
        if not self.configured():
            return []
        payload = {
            "organization_id": request.organization_id,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "query": request.query,
            "source_types": request.source_types,
            "top_k": request.top_k,
        }
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        try:
            if self._http is None:
                with build_http_client(self._settings) as client:
                    response = post_json_without_cookies(
                        client,
                        f"{self.base_url}/api/v1/internal/agent/retrieval/query",
                        payload=payload,
                        headers=headers,
                        timeout_sec=self.timeout_sec,
                    )
            else:
                response = post_json_without_cookies(
                    self._http,
                    f"{self.base_url}/api/v1/internal/agent/retrieval/query",
                    payload=payload,
                    headers=headers,
                    timeout_sec=self.timeout_sec,
                )
            response.raise_for_status()
        except httpx.PoolTimeout:
            metrics.inc("rag_runtime_go_bridge_pool_timeouts_total")
            logger.warning(
                "rag_runtime_go_bridge_pool_timeout",
                extra={"error_type": "pool_timeout"},
            )
            raise
        except httpx.HTTPError as exc:
            metrics.inc("rag_runtime_go_bridge_errors_total")
            logger.warning(
                "rag_runtime_go_bridge_http_error",
                extra={"error_type": type(exc).__name__},
            )
            raise
        raw: dict[str, Any] = response.json()
        chunks = raw.get("chunks", [])
        if not isinstance(chunks, list):
            return []
        return [ContextChunk.model_validate(item) for item in chunks if isinstance(item, dict)]
