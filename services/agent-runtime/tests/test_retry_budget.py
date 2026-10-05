"""Tests for ExecutionDeadline, ExecutionCancelled, and RetryBudget (Task 10)."""

from __future__ import annotations

import time


import pytest

from allcallall_agent_runtime.deadline import ExecutionCancelled, ExecutionDeadline, RetryBudget


# --------------------------------------------------------------------------- #
# ExecutionDeadline                                                            #
# --------------------------------------------------------------------------- #


class TestExecutionDeadlineFromHeader:
    """Deadline parsing from the X-AllCallAll-Deadline header (RFC3339Nano UTC)."""

    def test_valid_rfc3339_utc(self) -> None:
        """A valid RFC3339Nano UTC timestamp produces a non-expired deadline."""
        # 60 seconds from now in UTC
        future = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + 60))
        dl = ExecutionDeadline.from_header(future, default_seconds=120.0)
        assert not dl.expired()
        assert dl.remaining_seconds() > 50.0

    def test_valid_rfc3339_nano_utc(self) -> None:
        """A valid RFC3339Nano UTC timestamp (with fractional seconds) works."""
        future = time.strftime("%Y-%m-%dT%H:%M:%S.500000000+00:00", time.gmtime(time.time() + 30))
        dl = ExecutionDeadline.from_header(future, default_seconds=120.0)
        assert not dl.expired()
        assert dl.remaining_seconds() > 20.0

    def test_valid_rfc3339_z_suffix(self) -> None:
        """RFC3339 with Z suffix (common Go format) works."""
        future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60))
        dl = ExecutionDeadline.from_header(future, default_seconds=120.0)
        assert not dl.expired()
        assert dl.remaining_seconds() > 50.0

    def test_malformed_input_uses_default(self) -> None:
        """Malformed input falls back to the local default."""
        dl = ExecutionDeadline.from_header("not-a-timestamp", default_seconds=120.0)
        assert not dl.expired()
        assert dl.remaining_seconds() > 100.0

    def test_none_input_uses_default(self) -> None:
        """None header value falls back to the local default."""
        dl = ExecutionDeadline.from_header(None, default_seconds=120.0)
        assert not dl.expired()
        assert dl.remaining_seconds() > 100.0

    def test_already_expired_input(self) -> None:
        """A deadline in the past is immediately expired."""
        past = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 10))
        dl = ExecutionDeadline.from_header(past, default_seconds=120.0)
        assert dl.expired()

    def test_earlier_caller_deadline_wins(self) -> None:
        """When the caller deadline is earlier than the local max, the caller wins."""
        soon = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + 10))
        dl = ExecutionDeadline.from_header(soon, default_seconds=120.0)
        # The deadline should be close to 10 seconds, not 120
        assert dl.remaining_seconds() < 20.0
        assert dl.remaining_seconds() > 5.0

    def test_later_caller_deadline_capped_by_local_max(self) -> None:
        """When the caller deadline is later than the local max, the local max wins."""
        far = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + 600))
        dl = ExecutionDeadline.from_header(far, default_seconds=30.0)
        # The deadline should be capped at ~30 seconds, not 600
        assert dl.remaining_seconds() <= 35.0


class TestExecutionDeadlineCancel:
    """Cooperative cancellation via cancel() and raise_if_cancelled()."""

    def test_cancel_sets_cancelled(self) -> None:
        dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
        assert not dl.cancelled
        dl.cancel("client_cancelled")
        assert dl.cancelled

    def test_raise_if_cancelled_raises(self) -> None:
        dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
        dl.cancel("deadline_exceeded")
        with pytest.raises(ExecutionCancelled) as exc_info:
            dl.raise_if_cancelled()
        assert exc_info.value.reason == "deadline_exceeded"

    def test_raise_if_cancelled_noop_when_not_cancelled(self) -> None:
        dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
        dl.raise_if_cancelled()  # should not raise

    def test_expired_deadline_raises_on_check(self) -> None:
        past = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 5))
        dl = ExecutionDeadline.from_header(past, default_seconds=120.0)
        with pytest.raises(ExecutionCancelled) as exc_info:
            dl.raise_if_cancelled()
        assert exc_info.value.reason == "deadline_exceeded"

    def test_invalid_cancel_reason_rejected(self) -> None:
        dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
        with pytest.raises(ValueError):
            dl.cancel("invalid_reason")

    def test_valid_cancel_reasons(self) -> None:
        for reason in ("client_cancelled", "deadline_exceeded", "shutdown", "lease_lost"):
            dl = ExecutionDeadline.from_header(None, default_seconds=60.0)
            dl.cancel(reason)
            assert dl.cancelled
            with pytest.raises(ExecutionCancelled) as exc_info:
                dl.raise_if_cancelled()
            assert exc_info.value.reason == reason


class TestExecutionCancelled:
    def test_is_runtime_error(self) -> None:
        exc = ExecutionCancelled("shutdown")
        assert isinstance(exc, RuntimeError)

    def test_reason_attribute(self) -> None:
        exc = ExecutionCancelled("shutdown")
        assert exc.reason == "shutdown"


# --------------------------------------------------------------------------- #
# RetryBudget                                                                  #
# --------------------------------------------------------------------------- #


class _FakeClock:
    """Deterministic monotonic clock for testing RetryBudget."""

    def __init__(self, now: float = 100.0) -> None:
        self._now = now

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestRetryBudget:
    def test_allows_attempts_within_budget(self) -> None:
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=200.0, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)
        assert budget.attempts_used == 0
        delay = budget.next_delay(base=0.1, maximum=1.0)
        assert delay is not None
        assert delay == pytest.approx(0.1, abs=0.01)

    def test_stops_when_attempts_exhausted(self) -> None:
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=200.0, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=1, deadline=deadline, jitter=lambda _: 0.0)
        # First call consumes the attempt
        delay = budget.next_delay(base=0.1, maximum=1.0)
        assert delay is not None
        # Second call: budget exhausted
        delay2 = budget.next_delay(base=0.1, maximum=1.0)
        assert delay2 is None

    def test_stops_when_backoff_exceeds_remaining_time(self) -> None:
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=100.4, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)

        assert budget.next_delay(base=0.1, maximum=1.0) == 0.1
        clock.advance(0.35)
        assert budget.next_delay(base=0.1, maximum=1.0) is None

    def test_provider_and_workflow_share_budget(self) -> None:
        """Provider retries and outer workflow retries consume the same budget."""
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=105.0, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)

        # Provider uses 2 attempts
        d1 = budget.next_delay(base=0.1, maximum=1.0)
        assert d1 is not None
        d2 = budget.next_delay(base=0.1, maximum=1.0)
        assert d2 is not None

        # Only 1 attempt left in budget
        d3 = budget.next_delay(base=0.1, maximum=1.0)
        assert d3 is not None

        # Budget exhausted
        d4 = budget.next_delay(base=0.1, maximum=1.0)
        assert d4 is None

    def test_backoff_capped_by_maximum(self) -> None:
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=200.0, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=5, deadline=deadline, jitter=lambda _: 0.0)

        delays = []
        for _ in range(4):
            d = budget.next_delay(base=1.0, maximum=2.0)
            if d is None:
                break
            delays.append(d)
        # Exponential: 1.0, 2.0 (capped), 2.0 (capped), 2.0 (capped)
        assert delays[0] == pytest.approx(1.0, abs=0.01)
        assert all(d <= 2.0 for d in delays)

    def test_no_delay_when_deadline_expired(self) -> None:
        clock = _FakeClock(now=200.0)
        deadline = ExecutionDeadline(monotonic_deadline=100.0, clock=clock.monotonic)
        budget = RetryBudget(max_attempts=3, deadline=deadline, jitter=lambda _: 0.0)
        assert budget.next_delay(base=0.1, maximum=1.0) is None

    def test_remaining_seconds_decreases(self) -> None:
        clock = _FakeClock(now=100.0)
        deadline = ExecutionDeadline(monotonic_deadline=110.0, clock=clock.monotonic)
        r1 = deadline.remaining_seconds()
        clock.advance(5.0)
        r2 = deadline.remaining_seconds()
        assert r2 < r1
        assert r2 == pytest.approx(5.0, abs=0.1)
