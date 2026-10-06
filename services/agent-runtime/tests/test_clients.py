"""Lifecycle tests for process-lifetime HTTP client bundles (Task 11).

Proves that RuntimeClients reuses a single HTTP transport across multiple
calls, that owned clients close exactly once, and that injected clients
remain caller-owned.
"""

from __future__ import annotations

import httpx

from allcallall_agent_runtime.clients import build_runtime_clients
from allcallall_agent_runtime.config import AgentRuntimeConfig
from allcallall_agent_runtime.models import (
    RetrievalPlan,
    RetrievalPlanStep,
    WorkflowRequest,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _test_config(**overrides: object) -> AgentRuntimeConfig:
    defaults: dict[str, object] = dict(
        provider="openai_compatible",
        openai_base_url="http://test-provider",
        openai_api_key="test-key",
        openai_model="gpt-4",
        provider_strict=False,
        tool_bridge_base_url="http://test-tool-bridge",
        tool_bridge_token="test-token",
        rag_runtime_base_url="http://test-rag",
    )
    defaults.update(overrides)
    return AgentRuntimeConfig(**defaults)  # type: ignore[arg-type]


def _request() -> WorkflowRequest:
    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=9,
        goal="test goal",
    )


def _dummy_step() -> RetrievalPlanStep:
    return RetrievalPlanStep(step=1, source_scope="all", query="q", strategy="single_pass")


def _dummy_plan() -> RetrievalPlan:
    return RetrievalPlan(min_confidence=0.6, steps=[_dummy_step()])


# --------------------------------------------------------------------------- #
# Mock transports
# --------------------------------------------------------------------------- #

class CountingTransport(httpx.BaseTransport):
    """Transport that counts every request for testing connection reuse."""

    def __init__(self, *, response_json: dict | None = None) -> None:
        self._response_json = response_json or {}
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(200, json=self._response_json)


class RoutingTransport(httpx.BaseTransport):
    """Transport that routes responses by URL path and counts all requests."""

    def __init__(self) -> None:
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        url = str(request.url)
        if "/chat/completions" in url:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"summary": "ok", "action_items": []}'}}]},
            )
        if "/tools/read" in url:
            return httpx.Response(200, json={"output_json": "{}"})
        if "/retrieval/agentic" in url:
            return httpx.Response(
                200,
                json={
                    "evidence_pack": {"citations": [], "confidence": 0.9},
                    "context_sufficiency": {"sufficient": True},
                    "attempts": [{}],
                },
            )
        return httpx.Response(200, json={})


# --------------------------------------------------------------------------- #
# Tests — these will fail until clients.py is implemented
# --------------------------------------------------------------------------- #

def test_runtime_clients_reuse_one_transport_across_runs() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    clients.provider.synthesize(_request(), ["one"])
    clients.provider.synthesize(_request(), ["two"])

    assert transport.requests == 2


def test_runtime_clients_close_owned_client() -> None:
    clients = build_runtime_clients(_test_config())
    owned = clients._owned_http
    assert owned is not None
    clients.close()
    # After close, the owned reference is cleared
    assert clients._owned_http is None


def test_runtime_clients_does_not_close_injected_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    # Injected client is not owned
    assert clients._owned_http is None
    clients.close()
    # The injected client should still be usable
    clients.provider.synthesize(_request(), ["still works"])
    assert transport.requests == 1


def test_build_runtime_clients_creates_all_three() -> None:
    clients = build_runtime_clients(_test_config())
    assert clients.provider is not None
    assert clients.tool_bridge is not None
    assert clients.rag_runtime is not None
    clients.close()


def test_runtime_clients_provider_uses_shared_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    result = clients.provider.synthesize(_request(), ["test"])
    assert result is not None
    assert result.summary == "ok"
    assert transport.requests == 1


def test_runtime_clients_tool_bridge_uses_shared_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    bridge = clients.tool_bridge.build()
    obs = bridge.execute_read_tool(_request(), "query_context_chunks", {})
    assert obs is not None
    assert transport.requests == 1


def test_runtime_clients_rag_runtime_uses_shared_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    obs = clients.rag_runtime.agentic_retrieve(_request(), _dummy_step(), _dummy_plan())
    assert obs is not None
    assert transport.requests == 1


def test_close_is_idempotent() -> None:
    clients = build_runtime_clients(_test_config())
    clients.close()
    clients.close()  # second close should not raise
    assert clients._owned_http is None
