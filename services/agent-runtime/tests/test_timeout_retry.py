"""Tests for async retry primitive (P2#21) and request-level harness timeout (P2#21)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anyio
import pytest

import allcallall_agent_runtime.config as cfg
from allcallall_agent_runtime.checkpoint.store import NullCheckpointStore
from allcallall_agent_runtime.harness import AllCallAllAgentHarness, HarnessTimeoutExceeded
from allcallall_agent_runtime.retry import with_retry, with_retry_async


def _await(coro: Any) -> Any:
    async def run() -> Any:
        return await coro

    return anyio.run(run)


# --------------------------------------------------------------------------- #
# with_retry_async                                                             #
# --------------------------------------------------------------------------- #


def test_async_retry_succeeds_first_try() -> None:
    calls = 0

    async def attempt() -> int:
        nonlocal calls
        calls += 1
        return 7

    assert _await(with_retry_async(attempt, should_retry=lambda e: True)) == 7
    assert calls == 1


def test_async_retry_backoff_then_success() -> None:
    calls = 0

    async def attempt() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("transient")
        return "ok"

    result = _await(
        with_retry_async(attempt, should_retry=lambda e: isinstance(e, ConnectionError), max_attempts=4)
    )
    assert result == "ok"
    assert calls == 3


def test_async_retry_exhaustion_raises_last() -> None:
    async def attempt() -> int:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        _await(
            with_retry_async(attempt, should_retry=lambda e: isinstance(e, ConnectionError), max_attempts=2)
        )


def test_async_retry_non_retryable_propagates() -> None:
    async def attempt() -> int:
        raise ValueError("bad")

    with pytest.raises(ValueError):
        _await(with_retry_async(attempt, should_retry=lambda e: isinstance(e, ConnectionError)))


def test_async_retry_sync_hook_called() -> None:
    hooks = []

    async def attempt() -> int:
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        _await(
            with_retry_async(
                attempt,
                should_retry=lambda e: isinstance(e, ConnectionError),
                max_attempts=2,
                on_retry=lambda exc, attempt_no: hooks.append(attempt_no),
            )
        )
    assert hooks == [1]


def test_async_retry_async_hook_called() -> None:
    hooks: list[int] = []

    async def attempt() -> int:
        raise ConnectionError("x")

    async def on_retry(exc: Exception, attempt_no: int) -> None:
        hooks.append(attempt_no)

    with pytest.raises(ConnectionError):
        _await(
            with_retry_async(
                attempt,
                should_retry=lambda e: isinstance(e, ConnectionError),
                max_attempts=2,
                on_retry=on_retry,
            )
        )
    assert hooks == [1]


# --------------------------------------------------------------------------- #
# sync with_retry still works (regression)                                     #
# --------------------------------------------------------------------------- #


def test_sync_retry_still_functional() -> None:
    calls = 0

    def attempt() -> int:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ConnectionError("x")
        return 1

    assert with_retry(attempt, should_retry=lambda e: isinstance(e, ConnectionError), max_attempts=3) == 1
    assert calls == 2


# --------------------------------------------------------------------------- #
# Harness request timeout                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def timeout_setting() -> Iterator[float]:
    previous = cfg.config.request_timeout_seconds
    cfg.config.request_timeout_seconds = 0.05
    yield 0.05
    cfg.config.request_timeout_seconds = previous


class _SlowGraph:
    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        import time

        time.sleep(0.3)  # exceeds the 0.05s deadline but stays short for the test
        return {}


def test_harness_invoke_timeout_raises(timeout_setting: float) -> None:
    harness = AllCallAllAgentHarness(checkpoint_store=NullCheckpointStore())
    harness._graph = _SlowGraph()  # bypass graph build; trigger the slow invoke
    request = _minimal_request()
    with pytest.raises(HarnessTimeoutExceeded) as exc:
        harness.run_workflow(request)
    assert exc.value.timeout_seconds == timeout_setting


def test_harness_timeout_disabled_runs_to_completion() -> None:
    previous = cfg.config.request_timeout_seconds
    cfg.config.request_timeout_seconds = 0.0
    try:
        harness = AllCallAllAgentHarness(checkpoint_store=NullCheckpointStore())
        harness._graph = _SlowGraph()
        request = _minimal_request()
        # With the deadline disabled, the slow (2s) invoke runs to completion and
        # then fails downstream (empty graph result) rather than timing out.
        result = harness.run_workflow(request)
        assert result.status in {"failed", "ready", "requires_action"}
    finally:
        cfg.config.request_timeout_seconds = previous


def _minimal_request() -> Any:
    from allcallall_agent_runtime.models import WorkflowRequest

    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=4,
        goal="test",
        preset="meeting_brief",
    )
