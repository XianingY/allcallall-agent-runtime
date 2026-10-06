"""Tests for the bounded admission controller."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from allcallall_agent_runtime.admission import AdmissionController, AdmissionLease, AdmissionRejected
from allcallall_agent_runtime.config import AgentRuntimeConfig, effective_max_active_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class AcquireResult:
    """Mutable container for the result of a background acquire."""
    lease: AdmissionLease | None = None
    error: AdmissionRejected | None = None
    waiting: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def _start_acquire(
    controller: AdmissionController,
    organization_id: int,
    deadline: float | None = None,
) -> AcquireResult:
    """Start an acquire in a background thread and return control primitives."""
    result = AcquireResult()

    def _run() -> None:
        result.waiting.set()
        try:
            result.lease = controller.acquire(organization_id=organization_id, deadline=deadline)
        except AdmissionRejected as exc:
            result.error = exc
        finally:
            result.done.set()

    result.thread = threading.Thread(target=_run, daemon=True)
    result.thread.start()
    result.waiting.wait(timeout=2)
    return result


def _join_and_close(result: AcquireResult, timeout: float = 5.0) -> None:
    """Wait for a background acquire thread to finish and close the lease."""
    result.done.wait(timeout=timeout)
    if result.thread is not None:
        result.thread.join(timeout=1)
    if result.lease is not None:
        result.lease.close()


# ---------------------------------------------------------------------------
# Basic admission
# ---------------------------------------------------------------------------


class TestAdmissionBasic:
    def test_acquire_and_release(self) -> None:
        controller = AdmissionController(max_active=2, max_queued=2)
        lease = controller.acquire(organization_id=1)
        assert controller.active_count == 1
        lease.close()
        assert controller.active_count == 0

    def test_acquire_context_manager(self) -> None:
        controller = AdmissionController(max_active=2, max_queued=2)
        with controller.acquire(organization_id=1):
            assert controller.active_count == 1
        assert controller.active_count == 0

    def test_close_is_idempotent(self) -> None:
        controller = AdmissionController(max_active=2, max_queued=2)
        lease = controller.acquire(organization_id=1)
        lease.close()
        lease.close()  # second close is a no-op
        assert controller.active_count == 0

    def test_max_active_slots(self) -> None:
        controller = AdmissionController(max_active=2, max_queued=0)
        lease1 = controller.acquire(organization_id=1)
        lease2 = controller.acquire(organization_id=2)
        assert controller.active_count == 2
        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=3)
        assert exc.value.reason == "queue_full"
        lease1.close()
        lease2.close()


# ---------------------------------------------------------------------------
# Queue behavior
# ---------------------------------------------------------------------------


class TestAdmissionQueue:
    def test_queued_acquire_gets_slot_on_release(self) -> None:
        controller = AdmissionController(max_active=1, max_queued=2, max_queue_wait_seconds=5.0)
        active = controller.acquire(organization_id=1)
        assert controller.active_count == 1

        bg = _start_acquire(controller, organization_id=2)
        time.sleep(0.05)
        assert controller.queued_count >= 1

        # Release the active slot — the queued waiter should get it
        active.close()
        bg.done.wait(timeout=3)

        assert bg.error is None
        assert bg.lease is not None
        assert controller.active_count == 1
        bg.lease.close()

    def test_admission_rejects_when_active_and_queue_are_full(self) -> None:
        controller = AdmissionController(max_active=1, max_queued=1, max_queue_wait_seconds=5.0)
        active = controller.acquire(organization_id=1)

        bg = _start_acquire(controller, organization_id=2)
        time.sleep(0.05)

        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=3)
        assert exc.value.reason == "queue_full"
        assert exc.value.retry_after_seconds >= 1

        active.close()
        _join_and_close(bg)

    def test_queue_timeout_rejects(self) -> None:
        controller = AdmissionController(max_active=1, max_queued=2, max_queue_wait_seconds=0.1)
        active = controller.acquire(organization_id=1)

        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=2, deadline=time.monotonic() + 0.05)
        assert exc.value.reason == "timeout"

        active.close()

    def test_fifo_order(self) -> None:
        """Waiters are served in FIFO order."""
        controller = AdmissionController(max_active=1, max_queued=4, max_queue_wait_seconds=10.0)
        active = controller.acquire(organization_id=1)

        bgs = [_start_acquire(controller, organization_id=org_id) for org_id in [10, 20, 30]]

        # Release the active slot — first waiter should get it
        time.sleep(0.05)
        active.close()

        # First waiter should have been granted
        bgs[0].done.wait(timeout=3)
        assert bgs[0].error is None
        assert bgs[0].lease is not None

        # Release one at a time
        if bgs[0].lease is not None:
            bgs[0].lease.close()

        bgs[1].done.wait(timeout=3)
        assert bgs[1].error is None
        assert bgs[1].lease is not None

        if bgs[1].lease is not None:
            bgs[1].lease.close()

        bgs[2].done.wait(timeout=3)
        assert bgs[2].error is None
        assert bgs[2].lease is not None

        if bgs[2].lease is not None:
            bgs[2].lease.close()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestAdmissionCancellation:
    def test_queued_cancellation_via_timeout(self) -> None:
        """A waiter that times out is removed from the queue."""
        controller = AdmissionController(max_active=1, max_queued=2, max_queue_wait_seconds=10.0)
        active = controller.acquire(organization_id=1)

        # A waiter that times out is removed
        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=2, deadline=time.monotonic() + 0.05)
        assert exc.value.reason == "timeout"
        assert controller.queued_count == 0

        active.close()


# ---------------------------------------------------------------------------
# Organization fairness
# ---------------------------------------------------------------------------


class TestAdmissionOrgFairness:
    def test_org_quota_enforced(self) -> None:
        """An org cannot exceed its per-org active limit."""
        controller = AdmissionController(max_active=4, max_queued=2, max_active_per_org=1)
        lease1 = controller.acquire(organization_id=1)
        # Org 1 has 1 active, which equals max_active_per_org
        # A second request from org 1 should be queued and then timeout
        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=1, deadline=time.monotonic() + 0.05)
        assert exc.value.reason == "timeout"

        # But org 2 can still acquire
        lease2 = controller.acquire(organization_id=2)
        assert controller.active_count == 2

        lease1.close()
        lease2.close()

    def test_default_org_quota_is_half_max_active(self) -> None:
        """Default max_active_per_org is ceil(max_active / 2)."""
        controller = AdmissionController(max_active=4)
        assert controller._max_active_per_org == 2

        controller2 = AdmissionController(max_active=5)
        assert controller2._max_active_per_org == 3

        controller3 = AdmissionController(max_active=1)
        assert controller3._max_active_per_org == 1


# ---------------------------------------------------------------------------
# No early release on HTTP timeout
# ---------------------------------------------------------------------------


class TestAdmissionNoEarlyRelease:
    def test_http_timeout_does_not_release_lease(self) -> None:
        """When an HTTP caller times out, the lease is NOT released early.

        The graph execution owner releases the lease in ``finally``; the HTTP
        timeout path only requests cancellation. This test verifies that a
        lease held by a running workflow is not released when a second caller
        times out waiting in the queue.
        """
        controller = AdmissionController(max_active=1, max_queued=2, max_queue_wait_seconds=10.0)
        active = controller.acquire(organization_id=1)

        # A second caller times out waiting
        with pytest.raises(AdmissionRejected) as exc:
            controller.acquire(organization_id=2, deadline=time.monotonic() + 0.05)
        assert exc.value.reason == "timeout"

        # The active lease should still be held
        assert controller.active_count == 1

        # And a new caller should still be able to queue
        bg = _start_acquire(controller, organization_id=3)
        time.sleep(0.05)
        assert controller.queued_count >= 1

        active.close()
        _join_and_close(bg)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestEffectiveMaxActiveRuns:
    def test_default_returns_configured(self) -> None:
        cfg = AgentRuntimeConfig(max_active_runs=8)
        assert effective_max_active_runs(cfg) == 8

    def test_clamped_by_mysql_pool_size(self) -> None:
        cfg = AgentRuntimeConfig(
            max_active_runs=16,
            checkpoint_mysql_enabled=True,
            checkpoint_mysql_pool_size=4,
        )
        assert effective_max_active_runs(cfg) == 4

    def test_no_clamp_when_mysql_disabled(self) -> None:
        cfg = AgentRuntimeConfig(
            max_active_runs=16,
            checkpoint_mysql_enabled=False,
            checkpoint_mysql_pool_size=4,
        )
        assert effective_max_active_runs(cfg) == 16

    def test_no_clamp_when_configured_within_pool(self) -> None:
        cfg = AgentRuntimeConfig(
            max_active_runs=4,
            checkpoint_mysql_enabled=True,
            checkpoint_mysql_pool_size=8,
        )
        assert effective_max_active_runs(cfg) == 4

    def test_pool_size_zero_no_clamp(self) -> None:
        """A zero pool size should not cause division or clamp issues."""
        cfg = AgentRuntimeConfig(
            max_active_runs=8,
            checkpoint_mysql_enabled=True,
            checkpoint_mysql_pool_size=0,
        )
        assert effective_max_active_runs(cfg) == 8
