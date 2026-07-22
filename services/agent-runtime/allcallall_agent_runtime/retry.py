from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_retry(
    func: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay_sec: float = 0.5,
    max_delay_sec: float = 4.0,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """Run ``func`` with exponential backoff and jitter on retryable errors.

    The ``should_retry`` predicate decides whether a raised exception is worth
    retrying. Non-retryable exceptions propagate immediately, and once
    ``max_attempts`` is exhausted the last exception is re-raised unchanged.

    This is the single resilience primitive used by the LLM provider, the Go
    tool bridge, and the RAG runtime client so that transient downstream faults
    (timeouts, connection resets, HTTP 429/5xx) do not fail an otherwise valid
    workflow run.
    """
    if max_attempts < 1:
        max_attempts = 1
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - caller scopes retries via should_retry
            last_exc = exc
            if attempt >= max_attempts or not should_retry(exc):
                raise
            if on_retry is not None:
                try:
                    on_retry(exc, attempt)
                except Exception:
                    # Never let metrics/observation break the retry loop.
                    pass
            delay = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
            delay += random.uniform(0, max(delay * 0.25, base_delay_sec * 0.1))
            time.sleep(delay)
    # Unreachable: the final attempt always re-raises. Kept for type-checkers.
    assert last_exc is not None
    raise last_exc
