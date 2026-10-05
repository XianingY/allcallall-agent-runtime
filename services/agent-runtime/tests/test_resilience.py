from __future__ import annotations

import time
import httpx
import pytest
from typing import Any

import allcallall_agent_runtime.config as _cfg
from allcallall_agent_runtime.models import RetrievalPlan, RetrievalPlanStep, WorkflowRequest
from allcallall_agent_runtime.providers import ProviderError
from allcallall_agent_runtime.providers.openai_compatible import OpenAICompatibleProvider
from allcallall_agent_runtime.retry import with_retry
from allcallall_agent_runtime.tool_bridge import GoToolBridge, ToolBridgeError
from allcallall_agent_runtime.rag_runtime_client import RAGRuntimeClient


def _provider(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> OpenAICompatibleProvider:
    settings: dict[str, Any] = dict(
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


def _bridge(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> GoToolBridge:
    settings: dict[str, Any] = dict(
        tool_bridge_base_url="http://go",
        tool_bridge_token="t",
        tool_bridge_max_retries=2,
    )
    settings.update(overrides)
    monkeypatch.setattr(_cfg, "config", _cfg.AgentRuntimeConfig(**settings))
    return GoToolBridge()


def _rag(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> RAGRuntimeClient:
    settings: dict[str, Any] = dict(rag_runtime_base_url="http://rag", rag_runtime_max_retries=2)
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


def _dummy_request() -> WorkflowRequest:
    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=9,
        goal="g",
    )


def _dummy_step() -> RetrievalPlanStep:
    return RetrievalPlanStep(step=1, source_scope="all", query="q", strategy="single_pass")


def _dummy_plan() -> RetrievalPlan:
    return RetrievalPlan(min_confidence=0.6, steps=[_dummy_step()])


# --------------------------------------------------------------------------- #
# Task 10: RetryBudget integration with provider / tool bridge / RAG
# --------------------------------------------------------------------------- #


def test_with_retry_budget_limits_provider_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a RetryBudget is provided, with_retry stops when the budget is exhausted."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline, RetryBudget

    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        raise ProviderError("transient", retryable=True)

    deadline = ExecutionDeadline(monotonic_deadline=100.0)  # already expired
    budget = RetryBudget(max_attempts=2, deadline=deadline, jitter=lambda _: 0.0)

    with pytest.raises(ProviderError):
        with_retry(
            flaky,
            should_retry=lambda e: isinstance(e, ProviderError) and e.retryable,
            max_attempts=5,
            base_delay_sec=0,
            max_delay_sec=0,
            budget=budget,
        )
    # Budget is expired, so next_delay returns None immediately.
    # The first call runs, but no retries are allowed.
    assert calls["n"] == 1


def test_with_retry_budget_allows_retries_within_deadline() -> None:
    """When the budget has room, with_retry retries and eventually succeeds."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline, RetryBudget

    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("transient", retryable=True)
        return 42

    deadline = ExecutionDeadline(monotonic_deadline=time.monotonic() + 10.0)
    budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)

    result = with_retry(
        flaky,
        should_retry=lambda e: isinstance(e, ProviderError) and e.retryable,
        max_attempts=5,
        base_delay_sec=0.001,
        max_delay_sec=0.001,
        budget=budget,
    )
    assert result == 42
    assert calls["n"] == 3


def test_provider_and_outer_retries_share_budget() -> None:
    """Provider retries and outer workflow retries consume the same budget."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline, RetryBudget

    class _FakeClock:
        def __init__(self) -> None:
            self._now = 100.0
        def monotonic(self) -> float:
            return self._now
        def advance(self, s: float) -> None:
            self._now += s

    clock = _FakeClock()
    deadline = ExecutionDeadline(monotonic_deadline=105.0, clock=clock.monotonic)
    budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)

    # Simulate provider using 2 budget slots
    d1 = budget.next_delay(base=0.01, maximum=0.1)
    assert d1 is not None
    d2 = budget.next_delay(base=0.01, maximum=0.1)
    assert d2 is not None

    # Outer workflow tries to retry — only 1 slot left
    d3 = budget.next_delay(base=0.01, maximum=0.1)
    assert d3 is not None

    # Budget exhausted
    d4 = budget.next_delay(base=0.01, maximum=0.1)
    assert d4 is None


def test_execution_deadline_cooperative_cancellation_in_node() -> None:
    """Nodes that call raise_if_cancelled() propagate ExecutionCancelled."""
    from allcallall_agent_runtime.deadline import ExecutionCancelled, ExecutionDeadline, set_current_deadline

    deadline = ExecutionDeadline(monotonic_deadline=time.monotonic() + 60.0)
    deadline.cancel("client_cancelled")
    set_current_deadline(deadline)
    try:
        with pytest.raises(ExecutionCancelled) as exc_info:
            deadline.raise_if_cancelled()
        assert exc_info.value.reason == "client_cancelled"
    finally:
        set_current_deadline(None)
