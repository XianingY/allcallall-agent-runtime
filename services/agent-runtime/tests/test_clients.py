"""Lifecycle tests for process-lifetime HTTP client bundles (Task 11).

Proves that RuntimeClients reuses a single HTTP transport across multiple
calls, that owned clients close exactly once, and that injected clients
remain caller-owned.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

import pytest
from pydantic import ValidationError

from allcallall_agent_runtime.clients import build_runtime_clients
from allcallall_agent_runtime.config import AgentRuntimeConfig
from allcallall_agent_runtime.models import (
    RetrievalPlan,
    RetrievalPlanStep,
    WorkflowRequest,
)
from allcallall_agent_runtime.providers.base import create_provider


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

class RoutingTransport(httpx.BaseTransport):
    """Transport that routes responses by URL path and counts all requests."""

    def __init__(self, *, set_cookie: bool = False) -> None:
        self.requests = 0
        self.seen_requests: list[httpx.Request] = []
        self.set_cookie = set_cookie

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        self.seen_requests.append(request)
        url = str(request.url)
        response: httpx.Response
        if "/chat/completions" in url:
            response = httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"summary": "ok", "action_items": []}'}}]},
            )
        elif "/tools/read" in url:
            response = httpx.Response(200, json={"output_json": "{}"})
        elif "/retrieval/agentic" in url:
            response = httpx.Response(
                200,
                json={
                    "evidence_pack": {"citations": [], "confidence": 0.9},
                    "context_sufficiency": {"sufficient": True},
                    "attempts": [{}],
                },
            )
        else:
            response = httpx.Response(200, json={})
        if self.set_cookie and self.requests == 1:
            response.headers["Set-Cookie"] = "session=tenant-one; Path=/"
        return response


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
    owned = clients.owned_http
    assert owned is not None
    with patch.object(owned, "close", wraps=owned.close) as close:
        clients.close()
        clients.close()
        close.assert_called_once()
    assert clients.owned_http is None
    assert clients.closed is True


def test_runtime_clients_does_not_close_injected_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    # Injected client is not owned
    assert clients.owned_http is None
    with patch.object(http, "close", wraps=http.close) as close:
        clients.close()
        close.assert_not_called()
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


def test_runtime_clients_share_one_transport_across_all_services() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    clients.provider.synthesize(_request(), ["provider"])
    clients.tool_bridge.build().execute_read_tool(_request(), "lookup", {})
    clients.rag_runtime.agentic_retrieve(_request(), _dummy_step(), _dummy_plan())

    assert transport.requests == 3


def test_runtime_clients_do_not_replay_cookies_across_requests() -> None:
    transport = RoutingTransport(set_cookie=True)
    http = httpx.Client(transport=transport)
    clients = build_runtime_clients(_test_config(), http_client=http)

    clients.provider.synthesize(_request(), ["tenant one"])
    clients.provider.synthesize(_request(), ["tenant two"])

    assert transport.requests == 2
    assert all(request.headers.get("cookie") is None for request in transport.seen_requests)
    assert not http.cookies


def test_runtime_clients_preserve_service_timeouts_on_shared_client() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport, timeout=99)
    config = _test_config(
        openai_timeout_sec=1,
        tool_bridge_timeout_sec=2,
        rag_runtime_timeout_sec=3,
    )
    clients = build_runtime_clients(config, http_client=http)

    clients.provider.synthesize(_request(), [])
    clients.tool_bridge.build().execute_read_tool(_request(), "lookup", {})
    clients.rag_runtime.agentic_retrieve(_request(), _dummy_step(), _dummy_plan())

    expected_timeouts = [1, 2, 3]
    actual_timeouts = [request.extensions["timeout"]["read"] for request in transport.seen_requests]
    assert actual_timeouts == expected_timeouts


def test_close_is_idempotent() -> None:
    clients = build_runtime_clients(_test_config())
    clients.close()
    clients.close()  # second close should not raise
    assert clients.owned_http is None


def test_create_provider_accepts_injected_client_and_config() -> None:
    transport = RoutingTransport()
    http = httpx.Client(transport=transport)
    provider = create_provider(config=_test_config(), http_client=http)

    synthesis = provider.synthesize(_request(), [])

    assert synthesis is not None
    assert synthesis.summary == "ok"
    assert transport.requests == 1


def test_create_provider_default_client_is_bounded_and_cookieless() -> None:
    config = _test_config()
    with patch("httpx.Client", wraps=httpx.Client) as client_factory:
        provider = create_provider(config=config)
        client = provider._client

    assert client is not None
    kwargs = client_factory.call_args.kwargs
    assert kwargs["limits"].max_connections == config.http_max_connections
    assert kwargs["limits"].max_keepalive_connections == config.http_max_keepalive_connections
    assert kwargs["cookies"]._policy.set_ok(object(), object()) is False


@pytest.mark.parametrize(
    "field",
    [
        "http_max_connections",
        "http_max_keepalive_connections",
        "http_keepalive_expiry_sec",
        "http_connect_timeout_sec",
        "http_read_timeout_sec",
        "http_write_timeout_sec",
        "http_pool_timeout_sec",
    ],
)
def test_agent_http_pool_settings_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        _test_config(**{field: 0})


def test_agent_keepalive_limit_cannot_exceed_total_connections() -> None:
    with pytest.raises(ValidationError):
        _test_config(http_max_connections=2, http_max_keepalive_connections=3)
