"""Bounded admission control for workflow runs.

Limits the number of concurrently active workflow runs (``max_active_runs``) and
the number of queued waiters (``max_queued_runs``).  When both are full the
controller rejects the request with :class:`AdmissionRejected`.  A queued
waiter that exceeds ``max_queue_wait_seconds`` is also rejected.

The implementation uses ``threading.Condition`` with an explicit waiter deque
and monotonic deadlines so that cancellation is prompt and no lease is released
early when an HTTP caller times out.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from types import TracebackType
from typing import Optional

from .metrics import (
    admission_accepted_total,
    admission_active_runs,
    admission_queued_runs,
    admission_queue_wait_seconds,
    admission_rejected_total,
)

logger = logging.getLogger(__name__)


class AdmissionRejected(RuntimeError):
    """Raised when admission control cannot accept a workflow run."""

    def __init__(self, reason: str, retry_after_seconds: float) -> None:
        super().__init__(f"admission rejected: {reason}")
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


class AdmissionLease:
    """A lease representing one active workflow run slot.

    Call ``close()`` (or use as a context manager) to release the slot back to
    the admission controller.  ``close()`` is idempotent.
    """

    __slots__ = ("_controller", "_org_id", "_closed", "_lock")

    def __init__(self, controller: AdmissionController, org_id: int) -> None:
        self._controller = controller
        self._org_id = org_id
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._controller._release(self)

    def __enter__(self) -> AdmissionLease:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


class _Waiter:
    """A single queued admission request waiting for a slot."""

    __slots__ = ("event", "acquired", "lease", "cancelled", "org_id", "enqueued_at")

    def __init__(self, org_id: int) -> None:
        self.event = threading.Event()
        self.acquired = False
        self.lease: Optional[AdmissionLease] = None
        self.cancelled = False
        self.org_id = org_id
        self.enqueued_at = time.monotonic()


class AdmissionController:
    """Bounded admission controller with FIFO queue and per-org fairness.

    Uses ``threading.Condition`` for signaling and a deque of waiters for FIFO
    ordering.  Organization fairness is enforced by capping the share of active
    slots any single organization can hold (``max_active_per_org``), defaulting
    to half of ``max_active`` (rounded up) when not explicitly set.
    """

    def __init__(
        self,
        max_active: int = 4,
        max_queued: int = 16,
        max_queue_wait_seconds: float = 5.0,
        max_active_per_org: int | None = None,
    ) -> None:
        self._max_active = max_active
        self._max_queued = max_queued
        self._max_queue_wait_seconds = max_queue_wait_seconds
        self._max_active_per_org = max_active_per_org or max((max_active + 1) // 2, 1)
        self._active = 0
        self._org_active: dict[int, int] = {}
        self._condition = threading.Condition()
        self._waiters: deque[_Waiter] = deque()
        admission_active_runs.set(0)
        admission_queued_runs.set(0)

    @property
    def max_active(self) -> int:
        return self._max_active

    @property
    def max_queued(self) -> int:
        return self._max_queued

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def queued_count(self) -> int:
        return len(self._waiters)

    def acquire(
        self,
        organization_id: int,
        deadline: float | None = None,
    ) -> AdmissionLease:
        """Acquire a lease for one active workflow run.

        Args:
            organization_id: The organization requesting the run (used for
                per-org fairness).
            deadline: Absolute monotonic deadline; if the caller cannot be
                admitted before this time, :class:`AdmissionRejected` is raised.
                When ``None`` the controller's ``max_queue_wait_seconds`` is
                used.

        Returns:
            An :class:`AdmissionLease` that must be closed (or used as a
            context manager) to release the slot.

        Raises:
            AdmissionRejected: When the queue is full or the deadline expires.
        """
        effective_deadline = deadline if deadline is not None else time.monotonic() + self._max_queue_wait_seconds
        enqueue_time = time.monotonic()

        with self._condition:
            # Fast path: slot available and org quota not exceeded
            if self._can_grant(organization_id):
                return self._grant(organization_id)

            # Queue is full — reject immediately
            if len(self._waiters) >= self._max_queued:
                admission_rejected_total.labels(reason="queue_full").inc()
                raise AdmissionRejected(
                    reason="queue_full",
                    retry_after_seconds=max(1.0, self._max_queue_wait_seconds),
                )

            # Enqueue
            waiter = _Waiter(organization_id)
            self._waiters.append(waiter)
            admission_queued_runs.set(len(self._waiters))

        # Wait outside the condition lock for the event signal
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            self._remove_waiter(waiter)
            admission_rejected_total.labels(reason="timeout").inc()
            raise AdmissionRejected(reason="timeout", retry_after_seconds=1.0)

        waiter.event.wait(timeout=max(0, remaining))

        if waiter.cancelled:
            self._remove_waiter(waiter)
            admission_rejected_total.labels(reason="cancelled").inc()
            raise AdmissionRejected(reason="cancelled", retry_after_seconds=1.0)

        if waiter.acquired and waiter.lease is not None:
            wait_time = time.monotonic() - enqueue_time
            admission_queue_wait_seconds.observe(max(0.0, wait_time))
            return waiter.lease

        # Timed out
        self._remove_waiter(waiter)
        admission_rejected_total.labels(reason="timeout").inc()
        raise AdmissionRejected(reason="timeout", retry_after_seconds=1.0)

    def _can_grant(self, organization_id: int) -> bool:
        """Check if a lease can be granted (caller must hold ``_condition``)."""
        if self._active >= self._max_active:
            return False
        if self._org_active.get(organization_id, 0) >= self._max_active_per_org:
            return False
        return True

    def _grant(self, organization_id: int) -> AdmissionLease:
        """Grant a lease (caller must hold ``_condition``)."""
        lease = AdmissionLease(self, organization_id)
        self._active += 1
        self._org_active[organization_id] = self._org_active.get(organization_id, 0) + 1
        admission_active_runs.set(self._active)
        admission_accepted_total.inc()
        return lease

    def _release(self, lease: AdmissionLease) -> None:
        """Release a lease and wake the next eligible waiter in FIFO order."""
        with self._condition:
            self._active -= 1
            org_id = lease._org_id
            org_count = self._org_active.get(org_id, 0)
            if org_count <= 1:
                self._org_active.pop(org_id, None)
            else:
                self._org_active[org_id] = org_count - 1
            admission_active_runs.set(self._active)
            self._try_grant_waiters()

    def _remove_waiter(self, waiter: _Waiter) -> None:
        """Remove a waiter from the deque (best-effort)."""
        with self._condition:
            try:
                self._waiters.remove(waiter)
            except ValueError:
                pass
            admission_queued_runs.set(len(self._waiters))

    def _try_grant_waiters(self) -> None:
        """Try to grant leases to waiting callers (caller must hold ``_condition``)."""
        while self._waiters and self._active < self._max_active:
            # Scan for an eligible waiter (org quota not exceeded)
            granted = False
            for i, waiter in enumerate(self._waiters):
                if waiter.cancelled:
                    continue
                if self._org_active.get(waiter.org_id, 0) < self._max_active_per_org:
                    # Grant to this waiter
                    lease = AdmissionLease(self, waiter.org_id)
                    self._active += 1
                    self._org_active[waiter.org_id] = self._org_active.get(waiter.org_id, 0) + 1
                    admission_active_runs.set(self._active)
                    admission_accepted_total.inc()
                    admission_queued_runs.set(len(self._waiters) - 1)
                    waiter.acquired = True
                    waiter.lease = lease
                    del self._waiters[i]
                    waiter.event.set()
                    granted = True
                    break
            if not granted:
                # No eligible waiter found (all at org quota); stop trying
                break
        # Clean up cancelled waiters from the front
        while self._waiters and self._waiters[0].cancelled:
            self._waiters.popleft()
        admission_queued_runs.set(len(self._waiters))
