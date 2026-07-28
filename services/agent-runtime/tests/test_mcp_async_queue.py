"""Tests for MCP tool wrapping + asynchronous tool queue (Module 6)."""

from __future__ import annotations

from allcallall_agent_runtime.async_tool_queue import AsyncToolQueue
from allcallall_agent_runtime.mcp_tools import (
    EXEC_ASYNC,
    READ,
    WRITE,
    MARKDOWN_WRITE,
    MEETING_TRANSCRIBE,
    QUERY_CONTEXT,
    default_registry,
)


# --- MCP tool wrapping --------------------------------------------------- #

def test_mcp_tool_classification() -> None:
    assert QUERY_CONTEXT.kind == READ
    assert QUERY_CONTEXT.execution_mode == "sync"
    assert MARKDOWN_WRITE.kind == WRITE
    assert MARKDOWN_WRITE.execution_mode == EXEC_ASYNC
    assert MEETING_TRANSCRIBE.kind == WRITE
    assert MEETING_TRANSCRIBE.execution_mode == EXEC_ASYNC


def test_mcp_tool_to_mcp_shape() -> None:
    entry = QUERY_CONTEXT.to_mcp()
    assert entry["name"] == "query_context_chunks"
    assert entry["inputSchema"]["type"] == "object"
    assert entry["annotations"]["kind"] == READ


def test_registry_splits_read_write() -> None:
    reg = default_registry()
    names = {t["name"] for t in reg.tool_list()}
    assert {"query_context_chunks", "write_markdown_document", "transcribe_meeting_recording"} <= names
    assert {t.name for t in reg.read_tools()} == {"query_context_chunks"}
    assert {t.name for t in reg.write_tools()} == {"write_markdown_document", "transcribe_meeting_recording"}


# --- async queue: idempotent creation ------------------------------------ #

def test_enqueue_is_idempotent() -> None:
    q = AsyncToolQueue()
    t1 = q.enqueue("write_markdown_document", {"title": "x"}, idempotency_key="k1")
    t2 = q.enqueue("write_markdown_document", {"title": "y"}, idempotency_key="k1")
    assert t1 == t2  # duplicate key -> same task, no twin


# --- async queue: lease + completion ------------------------------------- #

def test_claim_grants_lease_and_completes() -> None:
    q = AsyncToolQueue()
    tid = q.enqueue("write_markdown_document", {}, idempotency_key="k1")
    task = q.claim("worker-1")
    assert task is not None and task.task_id == tid
    assert task.status == "processing"
    assert task.lease_owner == "worker-1"
    q.complete(tid)
    assert q.claim("worker-2") is None  # nothing left


def test_expired_lease_is_reclaimable() -> None:
    q = AsyncToolQueue()
    tid = q.enqueue("write_markdown_document", {}, idempotency_key="k1")
    q.claim("worker-1", now=100.0, visibility_timeout=10.0)
    # Lease expired at t=200; a new worker can reclaim.
    again = q.claim("worker-2", now=200.0, visibility_timeout=10.0)
    assert again is not None and again.task_id == tid
    assert again.lease_owner == "worker-2"


# --- async queue: traffic shaping (rate limit) --------------------------- #

def test_rate_limit_throttles_concurrent_per_key() -> None:
    q = AsyncToolQueue(rate_limit_max=1)
    q.enqueue("write_markdown_document", {}, idempotency_key="a", rate_limit_key="doc")
    q.enqueue("write_markdown_document", {}, idempotency_key="b", rate_limit_key="doc")
    first = q.claim("w1")
    assert first is not None
    # Second claim blocked: same rate_limit_key at concurrency cap 1.
    second = q.claim("w2")
    assert second is None
    # After completion, the key is free again.
    q.complete(first.task_id)
    second = q.claim("w2")
    assert second is not None


# --- async queue: retry with backoff + DLQ ------------------------------- #

def test_failure_retries_then_dead_letters() -> None:
    q = AsyncToolQueue(base_backoff_sec=1.0)
    tid = q.enqueue("write_markdown_document", {}, idempotency_key="k1", max_attempts=2)
    t0 = q.claim("w")
    assert t0 is not None
    q.fail(t0.task_id, "boom", now=0.0)
    # Re-queued with backoff; not claimable until scheduled_at.
    assert q.claim("w", now=0.0) is None
    retried = q.claim("w", now=10.0)
    assert retried is not None and retried.task_id == tid
    # Second failure exhausts attempts -> dead letter.
    q.fail(retried.task_id, "boom again", now=10.0)
    assert q.claim("w", now=20.0) is None
    dead = q.dead_letters()
    assert len(dead) == 1 and dead[0].task_id == tid and dead[0].status == "dead"


def test_priority_ordering() -> None:
    q = AsyncToolQueue()
    q.enqueue("write_markdown_document", {}, idempotency_key="low", priority=9)
    q.enqueue("write_markdown_document", {}, idempotency_key="high", priority=1)
    claimed = q.claim("w")
    assert claimed is not None
    assert claimed.idempotency_key == "high"  # lower number = higher priority
