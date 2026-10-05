"""Tests for async retry primitive (P2#21) and request-level harness timeout (P2#21)."""

from __future__ import annotations

import time
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



# --------------------------------------------------------------------------- #
# Invoke executor lifecycle (single-owner injection pattern)                  #
# --------------------------------------------------------------------------- #


def test_set_invoke_executor_injects_executor() -> None:
    """set_invoke_executor injects an executor that _get_invoke_executor returns."""
    import concurrent.futures
    from allcallall_agent_runtime.orchestration.harness import (
        _get_invoke_executor,
        set_invoke_executor,
        shutdown_invoke_executor,
    )

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-inject")
        set_invoke_executor(executor)
        assert _get_invoke_executor() is executor
    finally:
        shutdown_invoke_executor(wait=False)


def test_shutdown_invoke_executor_closes_injected() -> None:
    """shutdown_invoke_executor shuts down the injected executor."""
    import concurrent.futures
    from allcallall_agent_runtime.orchestration.harness import (
        _get_invoke_executor,
        set_invoke_executor,
        shutdown_invoke_executor,
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-shutdown")
    set_invoke_executor(executor)
    shutdown_invoke_executor(wait=False)

    # After shutdown, _get_invoke_executor lazily creates a new one
    new_executor = _get_invoke_executor()
    assert new_executor is not executor
    shutdown_invoke_executor(wait=False)


def test_invoke_executor_sized_from_config() -> None:
    """The invoke executor is sized from effective_max_active_runs."""
    from allcallall_agent_runtime.orchestration.harness import (
        _get_invoke_executor,
        shutdown_invoke_executor,
    )
    from allcallall_agent_runtime.config import effective_max_active_runs, config

    # Clean up any existing executor
    shutdown_invoke_executor(wait=False)

    expected = effective_max_active_runs(config)
    executor = _get_invoke_executor()
    assert executor._max_workers == expected

    # Clean up
    shutdown_invoke_executor(wait=False)


# --------------------------------------------------------------------------- #
# Task 10: ExecutionDeadline integration with harness and retry budget
# --------------------------------------------------------------------------- #


def test_execution_deadline_from_header_valid() -> None:
    """A valid RFC3339 deadline header produces a non-expired deadline."""
    import time as _time
    from allcallall_agent_runtime.deadline import ExecutionDeadline

    future = _time.strftime("%Y-%m-%dT%H:%M:%S+00:00", _time.gmtime(_time.time() + 60))
    dl = ExecutionDeadline.from_header(future, default_seconds=120.0)
    assert not dl.expired()
    assert dl.remaining_seconds() > 50.0


def test_execution_deadline_from_header_none_uses_default() -> None:
    """When the header is absent, the local default is used."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline

    dl = ExecutionDeadline.from_header(None, default_seconds=30.0)
    assert not dl.expired()
    assert dl.remaining_seconds() <= 35.0


def test_execution_deadline_cancel_and_raise() -> None:
    """cancel() + raise_if_cancelled() raises ExecutionCancelled."""
    from allcallall_agent_runtime.deadline import ExecutionCancelled, ExecutionDeadline

    dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
    dl.cancel("shutdown")
    with pytest.raises(ExecutionCancelled) as exc_info:
        dl.raise_if_cancelled()
    assert exc_info.value.reason == "shutdown"


def test_retry_budget_stops_when_backoff_exceeds_remaining() -> None:
    """RetryBudget returns None when the backoff delay would exceed the remaining deadline."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline, RetryBudget

    class _FakeClock:
        def __init__(self, now: float = 100.0) -> None:
            self._now = now
        def monotonic(self) -> float:
            return self._now
        def advance(self, seconds: float) -> None:
            self._now += seconds

    clock = _FakeClock(now=100.0)
    deadline = ExecutionDeadline(monotonic_deadline=100.4, clock=clock.monotonic)
    budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)

    assert budget.next_delay(base=0.1, maximum=1.0) == 0.1
    clock.advance(0.35)
    assert budget.next_delay(base=0.1, maximum=1.0) is None


def test_with_retry_respects_budget() -> None:
    """with_retry stops retrying when the retry budget says no more attempts."""
    from allcallall_agent_runtime.deadline import ExecutionDeadline, RetryBudget

    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        raise ConnectionError("transient")

    deadline = ExecutionDeadline(monotonic_deadline=time.monotonic() + 0.5)
    budget = RetryBudget(max_attempts=2, deadline=deadline, jitter=lambda _: 0.0)

    with pytest.raises(ConnectionError):
        with_retry(
            flaky,
            should_retry=lambda e: isinstance(e, ConnectionError),
            max_attempts=5,  # would allow 5, but budget limits to 2
            base_delay_sec=0.01,
            max_delay_sec=0.01,
            budget=budget,
        )
    # Budget allowed 2 attempts (next_delay consumed 2), but the first call
    # doesn't consume a budget slot — it's the retries that consume budget.
    # So we expect 3 calls: 1 initial + 2 retries (budget slots).
    assert calls["n"] == 3


def test_with_retry_no_budget_preserves_legacy() -> None:
    """with_retry without a budget preserves the legacy max_attempts behavior."""
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return 42

    result = with_retry(
        flaky,
        should_retry=lambda e: isinstance(e, ConnectionError),
        max_attempts=3,
        base_delay_sec=0,
        max_delay_sec=0,
    )
    assert result == 42
    assert calls["n"] == 3


def test_harness_cancellation_grace_config() -> None:
    """cancellation_grace_seconds is wired and has a reasonable default."""
    from allcallall_agent_runtime.config import config

    assert hasattr(config, "cancellation_grace_seconds")
    assert config.cancellation_grace_seconds == 2.0
