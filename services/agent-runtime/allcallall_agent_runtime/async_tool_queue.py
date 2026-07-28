"""Async tool task queue for post-approval write tool execution (Module 6).

Write tools (e.g. Markdown export, meeting transcription) must not block the
user's streaming reply. Once a human approves a ``ToolProposal``, the runtime
enqueues it here and returns immediately; a background worker later claims and
executes it.

The queue guarantees:

* **Idempotent creation** — enqueueing with a duplicate ``idempotency_key``
  returns the original task instead of creating a twin.
* **Task lease** — a worker *claims* a task and holds a lease
  (``visibility_timeout``); uncompleted leases expire and the task becomes
  claimable again, so a crashed worker cannot lose work.
* **Traffic shaping** — claims respect a per-``rate_limit_key`` concurrency
  cap (``rate_limit_max``), smoothing bursts.
* **Retry with backoff** — a failed task is re-queued with an exponential
  backoff up to ``max_attempts``.
* **Dead-letter queue** — tasks that exhaust their attempts land in the DLQ
  for inspection instead of looping forever.

The store is pluggable (:class:`TaskStore`); an in-memory implementation is
provided for tests and single-process deploys, mirroring the durable Go
Outbox foundation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class QueuedTask:
    task_id: str
    tool_name: str
    payload: dict[str, Any]
    idempotency_key: str
    queue_name: str = "default"
    priority: int = 5  # lower number = higher priority
    rate_limit_key: str = ""
    max_attempts: int = 3
    attempts: int = 0
    status: str = "queued"  # queued|processing|done|dead
    lease_owner: str = ""
    lease_until: float = 0.0
    scheduled_at: float = 0.0  # eligible for claim at/after this time (backoff)
    last_error: str = ""


@runtime_checkable
class TaskStore(Protocol):
    def add(self, task: QueuedTask) -> None:
        ...

    def get(self, task_id: str) -> QueuedTask | None:
        ...

    def all(self) -> list[QueuedTask]:
        ...

    def update(self, task: QueuedTask) -> None:
        ...

    def mark_dead(self, task: QueuedTask) -> None:
        ...


class InMemoryTaskStore:
    """Default single-process task store."""

    def __init__(self) -> None:
        self._tasks: dict[str, QueuedTask] = {}
        self._by_key: dict[str, str] = {}  # idempotency_key -> task_id
        self._dead: list[QueuedTask] = []

    def add(self, task: QueuedTask) -> None:
        self._tasks[task.task_id] = task
        self._by_key[task.idempotency_key] = task.task_id

    def get(self, task_id: str) -> QueuedTask | None:
        return self._tasks.get(task_id)

    def all(self) -> list[QueuedTask]:
        return list(self._tasks.values())

    def update(self, task: QueuedTask) -> None:
        self._tasks[task.task_id] = task

    def mark_dead(self, task: QueuedTask) -> None:
        self._dead.append(task)


class AsyncToolQueue:
    def __init__(
        self,
        store: TaskStore | None = None,
        rate_limit_max: int = 4,
        base_backoff_sec: float = 1.0,
    ) -> None:
        self._store = store or InMemoryTaskStore()
        self._rate_limit_max = max(1, rate_limit_max)
        self._base_backoff = max(0.0, base_backoff_sec)

    # ---- enqueue (idempotent) -------------------------------------------- #
    def enqueue(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        queue_name: str = "default",
        priority: int = 5,
        rate_limit_key: str = "",
        max_attempts: int = 3,
        execution_mode: str = "async_after_approval",
    ) -> str:
        del execution_mode  # recorded by the caller / proposal; queue only runs approved work
        existing_task_id = self._task_id_for_key(idempotency_key)
        existing = self._store.get(existing_task_id) if existing_task_id is not None else None
        if existing is not None:
            return existing.task_id  # idempotent: return the original task
        task = QueuedTask(
            task_id=uuid.uuid4().hex,
            tool_name=tool_name,
            payload=dict(payload),
            idempotency_key=idempotency_key,
            queue_name=queue_name,
            priority=priority,
            rate_limit_key=rate_limit_key,
            max_attempts=max_attempts,
        )
        self._store.add(task)
        return task.task_id

    def _task_id_for_key(self, key: str) -> str | None:
        # InMemoryTaskStore keeps the index; other stores implement get-by-key.
        store = self._store
        if isinstance(store, InMemoryTaskStore):
            return store._by_key.get(key)
        for task in store.all():
            if task.idempotency_key == key:
                return task.task_id
        return None

    # ---- claim (lease) --------------------------------------------------- #
    def claim(self, owner: str, now: float | None = None, visibility_timeout: float = 30.0) -> QueuedTask | None:
        now = now if now is not None else time.time()
        eligible = [
            t for t in self._store.all()
            if t.status in ("queued", "processing")
            and t.scheduled_at <= now
            and (t.lease_until <= now or t.lease_owner == "")  # not held by a live lease
        ]
        if not eligible:
            return None
        # Traffic shaping: respect per-key concurrency cap.
        in_flight: dict[str, int] = {}
        for t in self._store.all():
            if t.status == "processing":
                in_flight[t.rate_limit_key] = in_flight.get(t.rate_limit_key, 0) + 1
        # Highest priority (lowest number) first; stable by insertion order.
        eligible.sort(key=lambda t: t.priority)
        for task in eligible:
            key = task.rate_limit_key or ""
            if key and in_flight.get(key, 0) >= self._rate_limit_max:
                continue  # throttled: try next candidate
            task.status = "processing"
            task.lease_owner = owner
            task.lease_until = now + visibility_timeout
            self._store.update(task)
            return task
        return None

    # ---- completion / failure ------------------------------------------- #
    def complete(self, task_id: str) -> None:
        task = self._store.get(task_id)
        if task is None:
            return
        task.status = "done"
        task.lease_owner = ""
        task.lease_until = 0.0
        self._store.update(task)

    def fail(self, task_id: str, error: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        task = self._store.get(task_id)
        if task is None:
            return
        task.attempts += 1
        task.last_error = error
        if task.attempts >= task.max_attempts:
            task.status = "dead"
            task.lease_owner = ""
            task.lease_until = 0.0
            self._store.update(task)
            self._store.mark_dead(task)  # DLQ
            return
        # Re-queue with exponential backoff for the next attempt.
        backoff = self._base_backoff * (2 ** (task.attempts - 1))
        task.status = "queued"
        task.lease_owner = ""
        task.lease_until = 0.0
        task.scheduled_at = now + backoff
        self._store.update(task)

    def dead_letters(self) -> list[QueuedTask]:
        store = self._store
        if isinstance(store, InMemoryTaskStore):
            return list(store._dead)
        return [t for t in store.all() if t.status == "dead"]
