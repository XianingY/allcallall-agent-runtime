"""Optional Qdrant adapter for eval and specialized vector retrieval paths."""

from __future__ import annotations

from typing import Any

import httpx

from .config import config
from .models import ContextChunk, RetrievalQueryRequest


class QdrantAdapterError(RuntimeError):
    """Raised when the optional Qdrant adapter cannot complete a request."""


class QdrantAdapter:
    """HTTP adapter for Qdrant without making it a production data source of truth."""

    def __init__(self) -> None:
        self.url = config.qdrant_url.rstrip("/")
        self.collection = config.qdrant_collection
        self.api_key = config.qdrant_api_key
        self.timeout = config.qdrant_timeout_sec

    def configured(self) -> bool:
        return config.vector_store == "qdrant" and bool(self.url and self.collection)

    def query(self, request: RetrievalQueryRequest) -> list[ContextChunk]:
        if not self.configured():
            return []
        payload = self._search_payload(request) if request.query_vector else self._scroll_payload(request)
        endpoint = "search" if request.query_vector else "scroll"
        try:
            response = httpx.post(
                f"{self.url}/collections/{self.collection}/points/{endpoint}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise QdrantAdapterError(str(exc)) from exc
        if response.status_code >= 400:
            raise QdrantAdapterError(f"qdrant returned HTTP {response.status_code}")
        return self._parse_response(response.json(), vector_search=bool(request.query_vector))

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"api-key": self.api_key}

    def _search_payload(self, request: RetrievalQueryRequest) -> dict[str, Any]:
        return {
            "vector": request.query_vector,
            "limit": max(1, request.top_k),
            "with_payload": True,
            "filter": self._filter(request),
        }

    def _scroll_payload(self, request: RetrievalQueryRequest) -> dict[str, Any]:
        return {
            "limit": max(1, request.top_k),
            "with_payload": True,
            "filter": self._filter(request),
        }

    def _filter(self, request: RetrievalQueryRequest) -> dict[str, Any]:
        must: list[dict[str, Any]] = []
        if request.organization_id:
            must.append({"key": "organization_id", "match": {"value": request.organization_id}})
        if request.conversation_id:
            must.append({"key": "conversation_id", "match": {"value": request.conversation_id}})
        if request.source_types:
            must.append({"key": "source_type", "match": {"any": request.source_types}})
        return {"must": must} if must else {}

    def _parse_response(self, payload: dict[str, Any], *, vector_search: bool) -> list[ContextChunk]:
        result = payload.get("result", [])
        points = result if vector_search else result.get("points", []) if isinstance(result, dict) else []
        chunks: list[ContextChunk] = []
        for index, point in enumerate(points, start=1):
            if not isinstance(point, dict):
                continue
            point_payload = point.get("payload", {})
            if not isinstance(point_payload, dict):
                continue
            chunks.append(
                ContextChunk(
                    chunk_id=str(point_payload.get("chunk_id") or point.get("id", "")),
                    source_type=str(point_payload.get("source_type") or "knowledge"),
                    source_id=str(point_payload.get("source_id") or point.get("id", "")),
                    source_title=str(point_payload.get("source_title") or point_payload.get("title") or ""),
                    title=str(point_payload.get("title") or ""),
                    snippet=str(point_payload.get("snippet") or point_payload.get("text") or ""),
                    score=int(point_payload.get("score") or 0),
                    retrieval_mode="qdrant_vector" if vector_search else "qdrant_payload",
                    vector_rank=index if vector_search else 0,
                    vector_score=float(point.get("score") or 0),
                )
            )
        return chunks
