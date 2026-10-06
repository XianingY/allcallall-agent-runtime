from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

import allcallall_agent_runtime.config as _cfg
from allcallall_agent_runtime.config import AgentRuntimeConfig
from allcallall_agent_runtime.http_requests import build_http_client, post_json_without_cookies
from .metrics import registry
from .models import ContextChunk, WorkflowRequest
from .deadline import current_retry_budget
from .retry import with_retry


class ToolBridgeError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    input: dict[str, Any]
    output_json: str
    chunks: tuple[ContextChunk, ...] = ()


class GoToolBridge:
    def __init__(
        self,
        *,
        config: AgentRuntimeConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        settings = config or _cfg.config
        self._settings = settings
        self.base_url = settings.tool_bridge_base_url.strip().rstrip("/")
        self.token = settings.tool_bridge_token.strip()
        self.timeout_sec = max(1, int(settings.tool_bridge_timeout_sec))
        self.max_retries = max(0, int(settings.tool_bridge_max_retries))
        self._http = http_client

    @property
    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = build_http_client(self._settings)
        return self._http

    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def execute_read_tool(
        self,
        request: WorkflowRequest,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> ToolObservation | None:
        if not self.configured():
            return None
        payload = {
            "organization_id": request.organization_id,
            "user_id": request.user_id,
            "tool_name": tool_name,
            "arguments": tool_input,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        def _call() -> httpx.Response:
            try:
                response = post_json_without_cookies(
                    self._client,
                    f"{self.base_url}/api/v1/internal/agent/tools/read",
                    payload=payload,
                    headers=headers,
                    timeout_sec=self.timeout_sec,
                )
            except httpx.HTTPError as exc:
                raise ToolBridgeError(f"go tool bridge unavailable: {exc}", retryable=True) from exc
            if response.status_code == 429 or response.status_code >= 500:
                raise ToolBridgeError(
                    f"go tool bridge retryable status {response.status_code}", retryable=True
                )
            if response.status_code >= 400:
                raise ToolBridgeError(
                    f"go tool bridge returned {response.status_code}: {response.text[:300]}", retryable=False
                )
            return response

        # Only transient faults (network error, HTTP 429/5xx) are retried; a
        # 4xx from the Go backend is a permanent client/permission error.
        # When a request-scoped deadline is bound, retries consume the shared
        # budget so read-tool and workflow retries are bounded by the remaining
        # deadline.
        budget = current_retry_budget(max_attempts=self.max_retries + 1)
        response = with_retry(
            _call,
            should_retry=lambda exc: isinstance(exc, ToolBridgeError) and exc.retryable,
            max_attempts=self.max_retries + 1,
            base_delay_sec=self._settings.retry_base_delay_sec,
            max_delay_sec=self._settings.retry_max_delay_sec,
            budget=budget,
            on_retry=lambda exc, attempt: registry.counter(
                "agent_runtime_tool_bridge_retries_total",
                "Retries performed by the Go tool bridge client on transient faults",
            ).inc(),
        )
        body = response.json()
        output_json = str(body.get("output_json", ""))
        return ToolObservation(
            tool_name=tool_name,
            input=tool_input,
            output_json=output_json,
            chunks=tuple(chunks_from_tool_output(output_json)),
        )

    def execute_write_tool(
        self,
        *,
        organization_id: int,
        user_id: int,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Execute an approved write tool against the Go backend.

        Write tools (e.g. persisting a meeting brief) run *after* human approval,
        consumed from the async tool queue by the background worker. The request
        is authenticated with the shared bearer token, exactly like the read path.
        Raises :class:`ToolBridgeError` on failure (the worker then retries via the
        queue's bounded backoff / dead-letter policy).
        """
        if not self.configured():
            raise ToolBridgeError("go tool bridge is not configured", retryable=False)
        payload = {
            "organization_id": organization_id,
            "user_id": user_id,
            "tool_name": tool_name,
            "arguments": tool_input,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        def _call() -> httpx.Response:
            try:
                response = post_json_without_cookies(
                    self._client,
                    f"{self.base_url}/api/v1/internal/agent/tools/write",
                    payload=payload,
                    headers=headers,
                    timeout_sec=self.timeout_sec,
                )
            except httpx.HTTPError as exc:
                raise ToolBridgeError(f"go tool bridge unavailable: {exc}", retryable=True) from exc
            if response.status_code == 429 or response.status_code >= 500:
                raise ToolBridgeError(
                    f"go tool bridge retryable status {response.status_code}", retryable=True
                )
            if response.status_code >= 400:
                raise ToolBridgeError(
                    f"go tool bridge returned {response.status_code}: {response.text[:300]}", retryable=False
                )
            return response

        # Write operations execute exactly once at the HTTP transport level.
        # The async tool queue handles retries via its own bounded backoff and
        # dead-letter policy; retrying here would violate write-idempotency
        # expectations and risk duplicate side-effects.  Errors are classified
        # (retryable flag) so the queue worker can decide whether to re-enqueue.
        try:
            _call()
        except ToolBridgeError:
            raise


def chunks_from_tool_output(output_json: str) -> list[ContextChunk]:
    if not output_json.strip():
        return []
    try:
        payload = json.loads(output_json)
    except json.JSONDecodeError:
        return []
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return []
    out: list[ContextChunk] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        out.append(
            ContextChunk(
                chunk_id=str(item.get("chunk_id", "")),
                source_type=str(item.get("source_type", "")),
                source_id=str(item.get("source_id", "")),
                source_title=str(item.get("source_title", item.get("title", ""))),
                title=str(item.get("title", "")),
                snippet=str(item.get("snippet", "")),
                score=int_or_zero(item.get("score")),
                retrieval_mode=str(item.get("retrieval_mode", "")),
                bm25_rank=int_or_zero(item.get("bm25_rank")),
                vector_rank=int_or_zero(item.get("vector_rank")),
                rrf_score=float_or_zero(item.get("rrf_score")),
                bm25_score=float_or_zero(item.get("bm25_score")),
                vector_score=float_or_zero(item.get("vector_score")),
                rerank_score=float_or_zero(item.get("rerank_score")),
                rerank_reason=str(item.get("rerank_reason", "")),
                final_rank=int_or_zero(item.get("final_rank")),
                recording_session_id=optional_int(item.get("recording_session_id")),
                recording_file_id=optional_int(item.get("recording_file_id")),
                transcript_segment_id=optional_int(item.get("transcript_segment_id")),
                start_ms=optional_int(item.get("start_ms")),
                end_ms=optional_int(item.get("end_ms")),
            )
        )
    return out


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def int_or_zero(value: object) -> int:
    parsed = optional_int(value)
    return parsed or 0


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
