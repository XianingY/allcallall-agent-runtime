from __future__ import annotations

import httpx
import pytest

import allcallall_agent_runtime.config as _cfg
from allcallall_agent_runtime.providers import ProviderError
from allcallall_agent_runtime.providers.openai_compatible import OpenAICompatibleProvider
from allcallall_agent_runtime.retry import with_retry
from allcallall_agent_runtime.tool_bridge import GoToolBridge, ToolBridgeError
from allcallall_agent_runtime.rag_runtime_client import RAGRuntimeClient, RAGRuntimeError


def _provider(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> OpenAICompatibleProvider:
    settings = dict(
        provider="openai_compatible",
        openai_base_url="http://test",
        openai_api_key="k",
        openai_model="gpt-4",
        provider_strict=False,
        provider_max_retries=2,
    )
    settings.update(overrides)
    monkeypatch.setattr(_cfg, "config", _cfg.AgentRuntimeConfig(**settings))
    return OpenAICompatibleProvider()


def _bridge(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> GoToolBridge:
    settings = dict(
        tool_bridge_base_url="http://go",
        tool_bridge_token="t",
        tool_bridge_max_retries=2,
    )
    settings.update(overrides)
    monkeypatch.setattr(_cfg, "config", _cfg.AgentRuntimeConfig(**settings))
    return GoToolBridge()


def _rag(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> RAGRuntimeClient:
    settings = dict(rag_runtime_base_url="http://rag", rag_runtime_max_retries=2)
    settings.update(overrides)
    monkeypatch.setattr(_cfg, "config", _cfg.AgentRuntimeConfig(**settings))
    return RAGRuntimeClient()


def test_with_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("transient", retryable=True)
        return 42

    result = with_retry(
        flaky,
        should_retry=lambda e: isinstance(e, ProviderError) and e.retryable,
        max_attempts=3,
        base_delay_sec=0,
        max_delay_sec=0,
    )
    assert result == 42
    assert calls["n"] == 3


def test_with_retry_does_not_retry_permanent_errors() -> None:
    calls = {"n": 0}

    def permanent() -> int:
        calls["n"] += 1
        raise ProviderError("auth", retryable=False)

    with pytest.raises(ProviderError):
        with_retry(
            permanent,
            should_retry=lambda e: isinstance(e, ProviderError) and e.retryable,
            max_attempts=3,
            base_delay_sec=0,
            max_delay_sec=0,
        )
    assert calls["n"] == 1


def test_provider_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"summary": "ok", "action_items": []}'}}]},
        )

    provider = _provider(monkeypatch)
    provider._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    synthesis = provider.synthesize(_dummy_request(), [])
    assert synthesis is not None
    assert synthesis.summary == "ok"
    assert calls["n"] == 2


def test_provider_raises_immediately_on_401_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401)

    provider = _provider(monkeypatch)
    provider._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    with pytest.raises(ProviderError):
        provider.synthesize(_dummy_request(), [])
    assert calls["n"] == 1


def test_provider_exhausts_retries_on_500(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    provider = _provider(monkeypatch, provider_max_retries=2)
    provider._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    with pytest.raises(ProviderError):
        provider.synthesize(_dummy_request(), [])
    # max_attempts = retries + 1 = 3
    assert calls["n"] == 3


def test_tool_bridge_retries_network_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"output_json": "{}"})

    bridge = _bridge(monkeypatch)
    bridge._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    obs = bridge.execute_read_tool(_dummy_request(), "lookup", {})
    assert obs is not None
    assert calls["n"] == 2


def test_tool_bridge_4xx_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="forbidden")

    bridge = _bridge(monkeypatch)
    bridge._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    with pytest.raises(ToolBridgeError) as excinfo:
        bridge.execute_read_tool(_dummy_request(), "lookup", {})
    assert excinfo.value.retryable is False
    assert calls["n"] == 1


def test_rag_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"evidence_pack": {"citations": [], "confidence": 0.9}, "context_sufficiency": {"sufficient": True}, "attempts": [{}]},
        )

    rag = _rag(monkeypatch)
    rag._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    obs = rag.agentic_retrieve(_dummy_request(), _dummy_step(), _dummy_plan())
    assert obs is not None
    assert obs.sufficient is True
    assert calls["n"] == 2


def _dummy_request():
    from allcallall_agent_runtime.models import WorkflowRequest

    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=9,
        goal="g",
    )


def _dummy_step():
    from allcallall_agent_runtime.models import RetrievalPlanStep

    return RetrievalPlanStep(step=1, source_scope="all", query="q", strategy="single_pass")


def _dummy_plan():
    from allcallall_agent_runtime.models import RetrievalPlan

    return RetrievalPlan(min_confidence=0.6, steps=[_dummy_step()])
