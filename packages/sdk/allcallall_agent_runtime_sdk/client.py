from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from .models import (
    AgentRunRequest,
    AgentRunResponse,
    AgenticRetrievalRequest,
    AgenticRetrievalResponse,
    GroundingCheckRequest,
    GroundingCheckResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RerankRequest,
    RerankResponse,
    WorkflowRequest,
    WorkflowResponse,
)


class RuntimeClientError(RuntimeError):
    pass


def _payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return value


class _BaseClient:
    def __init__(self, base_url: str, timeout_sec: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.get(f"{self.base_url}{path}")
        return self._decode(response)

    def _post(self, path: str, payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(f"{self.base_url}{path}", json=_payload(payload))
        return self._decode(response)

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise RuntimeClientError(f"runtime returned {response.status_code}: {response.text[:500]}")
        raw = response.json()
        if not isinstance(raw, dict):
            raise RuntimeClientError("runtime returned a non-object JSON response")
        return raw


class AgentRuntimeClient(_BaseClient):
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def ready(self) -> dict[str, Any]:
        return self._get("/ready")

    def capabilities(self) -> dict[str, Any]:
        return self._get("/v1/capabilities")

    def run_react(self, request: AgentRunRequest | dict[str, Any]) -> AgentRunResponse:
        return AgentRunResponse.model_validate(self._post("/v1/agents/react/run", request))

    def run_workflow(self, preset: str, request: WorkflowRequest | dict[str, Any]) -> WorkflowResponse:
        return WorkflowResponse.model_validate(self._post(f"/v1/workflows/{preset}/run", request))

    def run_meeting_brief(self, request: WorkflowRequest | dict[str, Any]) -> WorkflowResponse:
        return self.run_workflow("meeting_brief", request)


class RAGRuntimeClient(_BaseClient):
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def ready(self) -> dict[str, Any]:
        return self._get("/ready")

    def query(self, request: RetrievalQueryRequest | dict[str, Any]) -> RetrievalQueryResponse:
        return RetrievalQueryResponse.model_validate(self._post("/v1/retrieval/query", request))

    def rerank(self, request: RerankRequest | dict[str, Any]) -> RerankResponse:
        return RerankResponse.model_validate(self._post("/v1/retrieval/rerank", request))

    def agentic_retrieve(
        self,
        request: AgenticRetrievalRequest | dict[str, Any],
    ) -> AgenticRetrievalResponse:
        return AgenticRetrievalResponse.model_validate(self._post("/v1/retrieval/agentic", request))

    def grounding_check(
        self,
        request: GroundingCheckRequest | dict[str, Any],
    ) -> GroundingCheckResponse:
        return GroundingCheckResponse.model_validate(self._post("/v1/grounding/check", request))

