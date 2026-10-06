from __future__ import annotations

import inspect
import random
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

import anyio

if TYPE_CHECKING:
    from .deadline import RetryBudget

T = TypeVar("T")


def with_retry(
    func: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay_sec: float = 0.5,
    max_delay_sec: float = 4.0,
    on_retry: Callable[[Exception, int], None] | None = None,
    budget: RetryBudget | None = None,
) -> T:
    """Run ``func`` with exponential backoff and jitter on retryable errors.

    The ``should_retry`` predicate decides whether a raised exception is worth
    retrying. Non-retryable exceptions propagate immediately, and once
    ``max_attempts`` is exhausted the last exception is re-raised unchanged.

    When a ``budget`` is provided, retries consume the shared budget and are
    only attempted when the backoff delay fits inside the remaining deadline.
    This lets provider and workflow retries share one budget that respects the
    caller-specified execution deadline.  When no budget is provided, the
    legacy behaviour (fixed ``max_attempts`` with no deadline awareness) is
    preserved.

    This is the single resilience primitive used by the LLM provider, the Go
    tool bridge, and the RAG runtime client so that transient downstream faults
    (timeouts, connection resets, HTTP 429/5xx) do not fail an otherwise valid
    workflow run.
    """
    if max_attempts < 1:
        max_attempts = 1
    last_exc: Exception | None = None
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - caller scopes retries via should_retry
            last_exc = exc
            if attempt >= max_attempts or not should_retry(exc):
                raise
            # Check budget: if a budget is provided, ask it for the next delay.
            # If the budget says no retry is possible (None), re-raise.
            if budget is not None:
                delay = budget.next_delay(base_delay_sec, max_delay_sec)
                if delay is None:
                    raise
            else:
                delay = _backoff_delay(attempt, base_delay_sec, max_delay_sec)
            if on_retry is not None:
                try:
                    on_retry(exc, attempt)
                except Exception:
                    # Never let metrics/observation break the retry loop.
                    pass
            time.sleep(delay)
    # Unreachable: the final attempt always re-raises. Kept for type-checkers.
    assert last_exc is not None
    raise last_exc


async def with_retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay_sec: float = 0.5,
    max_delay_sec: float = 4.0,
    on_retry: Callable[[Exception, int], None | Awaitable[None]] | None = None,
    budget: RetryBudget | None = None,
) -> T:
    """Async twin of :func:`with_retry`.

    Uses ``await`` for the wrapped callable and ``anyio.sleep`` for the
    backoff, so it never blocks the event loop. The ``on_retry`` hook may be a
    plain callable or a coroutine function. The synchronous primitive is kept
    intact for the non-async call paths (provider, tool bridge client).

    When a ``budget`` is provided, retries consume the shared budget and are
    only attempted when the backoff delay fits inside the remaining deadline.
    """
    if max_attempts < 1:
        max_attempts = 1
    last_exc: Exception | None = None
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - caller scopes retries via should_retry
            last_exc = exc
            if attempt >= max_attempts or not should_retry(exc):
                raise
            # Check budget: if a budget is provided, ask it for the next delay.
            # If the budget says no retry is possible (None), re-raise.
            if budget is not None:
                delay = budget.next_delay(base_delay_sec, max_delay_sec)
                if delay is None:
                    raise
            else:
                delay = _backoff_delay(attempt, base_delay_sec, max_delay_sec)
            if on_retry is not None:
                try:
                    hook = on_retry(exc, attempt)
                    if inspect.isawaitable(hook):
                        await hook
                except Exception:
                    # Never let metrics/observation break the retry loop.
                    pass
            await anyio.sleep(delay)
    # Unreachable: the final attempt always re-raises. Kept for type-checkers.
    assert last_exc is not None
    raise last_exc


def _backoff_delay(attempt: int, base_delay_sec: float, max_delay_sec: float) -> float:
    delay: float = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
    delay += random.uniform(0, max(delay * 0.25, base_delay_sec * 0.1))
    return delay
