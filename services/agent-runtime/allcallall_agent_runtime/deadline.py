"""Execution deadline, cooperative cancellation, and retry budget.

Provides the runtime primitives for propagating caller-specified deadlines
from the Go backend (via ``X-AllCallAll-Deadline`` / ``X-AllCallAll-Attempt``
headers) into the Python graph, enforcing cooperative cancellation at every
expensive operation, and bounding retries so that provider and workflow retries
share a single budget that must fit inside the remaining deadline.

Cancellation reasons are bounded to:
  - ``client_cancelled``  — HTTP caller disconnected / explicit cancel.
  - ``deadline_exceeded`` — The propagated deadline expired.
  - ``shutdown``          — Process is shutting down.
  - ``lease_lost``        — Admission lease was revoked.

The request-scoped deadline is propagated via a module-level context variable
(:func:`get_current_deadline`), not via LangGraph state keys, so it is
naturally excluded from checkpoint serialization and a resumed run starts
with a fresh deadline.
"""

from __future__ import annotations

import contextvars
import random
import time
from datetime import datetime, timezone
from typing import Callable

# Bounded set of valid cancellation reasons.
_VALID_CANCEL_REASONS = frozenset({"client_cancelled", "deadline_exceeded", "shutdown", "lease_lost"})


class ExecutionCancelled(RuntimeError):
    """Raised when an execution deadline is cancelled or expired.

    Carries a bounded reason code so observability can distinguish the cause
    without free-form strings.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        if reason not in _VALID_CANCEL_REASONS:
            raise ValueError(f"invalid cancellation reason: {reason!r}; expected one of {_VALID_CANCEL_REASONS}")
        super().__init__(f"execution cancelled: {reason}")
        self.reason = reason


class ExecutionDeadline:
    """A monotonic deadline for a single workflow run.

    Internally uses ``time.monotonic()`` so it is immune to clock adjustments.
    Absolute UTC timestamps are only used for header propagation and logging.

    Call ``raise_if_cancelled()`` at cooperative cancellation points (before
    and after expensive operations, at loop iterations).  Call ``cancel(reason)``
    to signal external cancellation (e.g. HTTP timeout, shutdown).

    The ``from_header`` class method parses the ``X-AllCallAll-Deadline`` header
    (RFC3339Nano UTC) and clamps the effective deadline to the local maximum
    (``default_seconds``).  When the header is absent or malformed, the local
    maximum is used as the deadline.
    """

    __slots__ = ("_monotonic_deadline", "_clock", "_cancelled", "_cancel_reason")

    def __init__(
        self,
        monotonic_deadline: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._monotonic_deadline = monotonic_deadline
        self._clock = clock or time.monotonic
        self._cancelled = False
        self._cancel_reason: str | None = None

    # ------------------------------------------------------------------ #
    # Construction from header
    # ------------------------------------------------------------------ #

    @classmethod
    def from_header(cls, value: str | None, default_seconds: float) -> ExecutionDeadline:
        """Parse an ``X-AllCallAll-Deadline`` header value.

        Args:
            value: The raw header value (RFC3339Nano UTC), or ``None`` when
                the header is absent.
            default_seconds: Local maximum deadline in seconds.  When the
                parsed caller deadline is later than this, it is capped.
                When the header is absent or malformed, this becomes the
                effective deadline.

        Returns:
            An :class:`ExecutionDeadline` with the effective monotonic
            deadline set.
        """
        caller_deadline_utc: datetime | None = None
        if value is not None:
            caller_deadline_utc = _parse_rfc3339(value)

        now_mono = time.monotonic()
        local_max_mono = now_mono + default_seconds

        if caller_deadline_utc is not None:
            # Convert the UTC wall-clock deadline to a monotonic deadline.
            # We compute the offset from "now UTC" to the caller deadline UTC,
            # then add that offset to the monotonic now.
            now_utc = datetime.now(timezone.utc)
            caller_offset = (caller_deadline_utc - now_utc).total_seconds()
            caller_mono = now_mono + caller_offset
            # Clamp: earlier caller deadline wins, later is capped by local max.
            effective_mono = min(caller_mono, local_max_mono)
        else:
            effective_mono = local_max_mono

        return cls(monotonic_deadline=effective_mono)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def monotonic_deadline(self) -> float:
        """The absolute monotonic deadline timestamp."""
        return self._monotonic_deadline

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    @property
    def cancel_reason(self) -> str | None:
        """The reason code passed to ``cancel()``, or ``None``."""
        return self._cancel_reason

    def remaining_seconds(self) -> float:
        """Seconds remaining until the deadline, clamped to >= 0."""
        return max(0.0, self._monotonic_deadline - self._clock())

    def expired(self) -> bool:
        """Whether the deadline has passed."""
        return self.remaining_seconds() <= 0.0

    def cancel(self, reason: str) -> None:
        """Request cooperative cancellation.

        Args:
            reason: One of the bounded cancellation reason codes.

        Raises:
            ValueError: If the reason is not in the valid set.
        """
        if reason not in _VALID_CANCEL_REASONS:
            raise ValueError(f"invalid cancellation reason: {reason!r}; expected one of {_VALID_CANCEL_REASONS}")
        self._cancelled = True
        self._cancel_reason = reason

    def raise_if_cancelled(self) -> None:
        """Raise :class:`ExecutionCancelled` if cancelled or expired.

        Called at cooperative cancellation points throughout the graph.
        When the deadline has expired, the reason is ``deadline_exceeded``.
        When cancellation was requested via ``cancel()``, the caller-supplied
        reason is used.
        """
        if self._cancelled:
            raise ExecutionCancelled(self._cancel_reason or "client_cancelled")
        if self.expired():
            raise ExecutionCancelled("deadline_exceeded")


class RetryBudget:
    """A shared retry budget that bounds both provider and workflow retries.

    Provider calls and outer workflow retries consume the same ``max_attempts``
    budget.  Each call to ``next_delay`` checks whether another attempt plus
    the backoff delay would fit inside the remaining deadline; if not, it
    returns ``None`` signalling that no further retries are possible.

    The ``jitter`` parameter is injectable for deterministic testing.
    When ``None``, a default jitter function adds up to 25% randomness.
    The jitter function receives the raw exponential delay and returns the
    *additional* jitter to add (not the final delay).
    """

    __slots__ = ("max_attempts", "_deadline", "_attempts_used", "_jitter_fn")

    def __init__(
        self,
        max_attempts: int,
        deadline: ExecutionDeadline,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self._deadline = deadline
        self._attempts_used = 0
        # jitter receives the raw exponential delay and returns the amount of
        # jitter to add.  Pass lambda _: 0.0 for deterministic testing.
        self._jitter_fn = jitter if jitter is not None else _default_jitter

    @property
    def attempts_used(self) -> int:
        """Number of retry attempts consumed so far."""
        return self._attempts_used

    @property
    def deadline(self) -> ExecutionDeadline:
        """The execution deadline governing this budget."""
        return self._deadline

    def next_delay(self, base: float, maximum: float) -> float | None:
        """Compute the backoff delay for the next retry attempt.

        Returns ``None`` when:
          - the budget is exhausted (``attempts_used >= max_attempts``), or
          - the deadline is expired, or
          - the computed backoff would exceed the remaining deadline.

        When a delay is returned, one attempt is consumed from the budget.
        The caller must sleep for the returned duration before retrying.

        Args:
            base: Base delay in seconds (first retry).
            maximum: Maximum delay cap in seconds.

        Returns:
            The backoff delay in seconds, or ``None`` if no retry is possible.
        """
        if self._attempts_used >= self.max_attempts:
            return None
        remaining = self._deadline.remaining_seconds()
        if remaining <= 0:
            return None

        # Exponential backoff: base * 2^(attempt-1), capped at maximum.
        attempt = self._attempts_used + 1
        raw_delay = min(maximum, base * (2 ** (attempt - 1)))
        # Add jitter: the jitter function receives the raw delay and returns
        # the additional jitter amount to add.
        jitter_amount: float = self._jitter_fn(raw_delay)
        delay = raw_delay + jitter_amount
        delay = min(delay, maximum)

        # The delay must fit inside the remaining deadline.  The next attempt
        # still needs time to execute, so we require delay < remaining.
        if delay >= remaining:
            return None

        self._attempts_used += 1
        return delay  # type: ignore[no-any-return]


def _default_jitter(raw_delay: float) -> float:
    """Return a random jitter amount (up to 25% of the raw delay)."""
    return random.uniform(0, max(raw_delay * 0.25, 0.01))


def _parse_rfc3339(value: str) -> datetime | None:
    """Best-effort RFC3339 / RFC3339Nano parser.

    Handles the common formats produced by Go's ``time.RFC3339Nano``:
      - ``2006-01-02T15:04:05Z``
      - ``2006-01-02T15:04:05.999999999Z``
      - ``2006-01-02T15:04:05+00:00``
      - ``2006-01-02T15:04:05.999999999+00:00``

    Returns ``None`` on parse failure (caller falls back to default).
    """
    try:
        # Python 3.11+ fromisoformat handles most RFC3339 variants including
        # fractional seconds and Z suffix, but we normalize Z -> +00:00
        # for compatibility with older Python versions.
        normalized = value.rstrip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        # Ensure timezone-aware UTC
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Convenience: derive a RetryBudget from the current request-scoped deadline
# --------------------------------------------------------------------------- #


def current_retry_budget(max_attempts: int) -> RetryBudget | None:
    """Derive a :class:`RetryBudget` from the current execution deadline.

    Returns ``None`` when no deadline is bound (standalone / test usage),
    preserving legacy retry behaviour.  When a deadline is bound, the budget
    is scoped to the remaining time so that provider and workflow retries
    share one pool and only retry when another attempt fits inside the
    remaining deadline.

    Args:
        max_attempts: Maximum retry attempts (typically from config, e.g.
            ``provider_max_retries + 1``).

    Returns:
        A :class:`RetryBudget` scoped to the current deadline, or ``None``.
    """
    deadline = get_current_deadline()
    if deadline is None:
        return None
    return RetryBudget(max_attempts=max_attempts, deadline=deadline)


# --------------------------------------------------------------------------- #
# Request-scoped context variable for graph runtime access
# --------------------------------------------------------------------------- #


_current_deadline: contextvars.ContextVar[ExecutionDeadline | None] = contextvars.ContextVar(
    "_current_deadline", default=None
)


def get_current_deadline() -> ExecutionDeadline | None:
    """Return the execution deadline for the current request scope.

    Set by the harness before graph invocation; read by nodes and retry
    helpers to check cancellation and compute remaining budget.  Returns
    ``None`` when no deadline is active (e.g. in tests or standalone usage).
    """
    return _current_deadline.get()


def set_current_deadline(deadline: ExecutionDeadline | None) -> None:
    """Bind or clear the execution deadline for the current request scope."""
    if deadline is not None and deadline.remaining_seconds() <= 0:
        import logging
        logging.getLogger(__name__).debug(
            "execution deadline is already expired or unbound (remaining=%.1fs); "
            "cooperative cancellation will fire immediately",
            deadline.remaining_seconds(),
        )
    _current_deadline.set(deadline)


def require_current_deadline() -> ExecutionDeadline:
    """Return the current deadline, raising if none is bound.

    Useful in nodes that require a deadline to be present.
    """
    dl = _current_deadline.get()
    if dl is None:
        raise RuntimeError("no execution deadline bound in current scope")
    return dl
