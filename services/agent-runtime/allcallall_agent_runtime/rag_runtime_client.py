from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

import allcallall_agent_runtime.config as _cfg
from allcallall_agent_runtime.metrics import registry
from allcallall_agent_runtime.models import ContextChunk, RetrievalPlan, RetrievalPlanStep, WorkflowRequest
from allcallall_agent_runtime.retry import with_retry


class RAGRuntimeError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class RAGRuntimeObservation:
    chunks: tuple[ContextChunk, ...]
    confidence: float
    sufficient: bool
    attempts: int


class RAGRuntimeClient:
    def __init__(self) -> None:
        self.base_url = _cfg.config.rag_runtime_base_url.strip().rstrip("/")
        self.timeout_sec = max(1, int(_cfg.config.rag_runtime_timeout_sec))
        self.max_retries = max(0, int(_cfg.config.rag_runtime_max_retries))
        self._http: httpx.Client | None = None

    @property
    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout_sec)
        return self._http

    def configured(self) -> bool:
        return bool(self.base_url)

    def agentic_retrieve(
        self,
        request: WorkflowRequest,
        step: RetrievalPlanStep,
        plan: RetrievalPlan,
    ) -> RAGRuntimeObservation | None:
        if not self.configured():
            return None
        source_types = [] if step.source_scope == "all" else [step.source_scope]
        payload = {
            "organization_id": request.organization_id,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "query": step.query,
            "source_types": source_types,
            "top_k": 6,
            "max_steps": 1,
            "min_confidence": plan.min_confidence,
            "chunks": [chunk.model_dump(exclude_none=True) for chunk in request.context_chunks],
        }

        def _call() -> httpx.Response:
            try:
                response = self._client.post(f"{self.base_url}/v1/retrieval/agentic", json=payload)
            except httpx.HTTPError as exc:
                raise RAGRuntimeError(f"rag runtime unavailable: {exc}", retryable=True) from exc
            if response.status_code == 429 or response.status_code >= 500:
                raise RAGRuntimeError(f"rag runtime retryable status {response.status_code}", retryable=True)
            if response.status_code >= 400:
                raise RAGRuntimeError(
                    f"rag runtime returned {response.status_code}: {response.text[:300]}", retryable=False
                )
            return response

        # Only transient faults (network error, HTTP 429/5xx) are retried; a 4xx
        # from the RAG runtime is a permanent request error.
        response = with_retry(
            _call,
            should_retry=lambda exc: isinstance(exc, RAGRuntimeError) and exc.retryable,
            max_attempts=self.max_retries + 1,
            base_delay_sec=_cfg.config.retry_base_delay_sec,
            max_delay_sec=_cfg.config.retry_max_delay_sec,
            on_retry=lambda exc, attempt: registry.counter(
                "agent_runtime_rag_retries_total",
                "Retries performed by the RAG runtime client on transient faults",
            ).inc(),
        )
        raw: dict[str, Any] = response.json()
        pack = raw.get("evidence_pack", {})
        sufficiency = raw.get("context_sufficiency", {})
        attempts = raw.get("attempts", [])
        citations = []
        if isinstance(pack, dict):
            raw_citations = pack.get("citations", [])
            if isinstance(raw_citations, list):
                citations = raw_citations
        chunks = tuple(
            ContextChunk.model_validate(item)
            for item in citations
            if isinstance(item, dict) and item.get("source_type") and item.get("snippet")
        )
        confidence = float_or_zero(pack.get("confidence") if isinstance(pack, dict) else None)
        sufficient = bool(sufficiency.get("sufficient")) if isinstance(sufficiency, dict) else False
        return RAGRuntimeObservation(
            chunks=chunks,
            confidence=confidence,
            sufficient=sufficient,
            attempts=len(attempts) if isinstance(attempts, list) else 0,
        )


def float_or_zero(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
