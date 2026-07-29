"""Tests for async tool-queue integration (P0#2) and skill manifest hardening (P2#22)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from fastapi.testclient import TestClient

import allcallall_agent_runtime.config as cfg
from allcallall_agent_runtime.async_tool_queue import AsyncToolQueue, ToolQueueWorker, get_default_tool_queue
from allcallall_agent_runtime.checkpoint.store import NullCheckpointStore
from allcallall_agent_runtime.harness import AllCallAllAgentHarness
from allcallall_agent_runtime.main import app
from allcallall_agent_runtime.models import ToolProposal, WorkflowRequest


def _request() -> WorkflowRequest:
    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=4,
        goal="g",
        preset="meeting_brief",
    )


def _proposal() -> ToolProposal:
    return ToolProposal(
        tool_name="write_conversation_message",
        arguments={"body": "hi"},
        idempotency_key="k1",
        priority="high",
        rate_limit_key="r",
        max_attempts=2,
    )


def test_harness_enqueues_proposals_when_queue_enabled() -> None:
    previous = cfg.config.enable_tool_queue
    cfg.config.enable_tool_queue = True
    try:
        queue: AsyncToolQueue = AsyncToolQueue()
        harness = AllCallAllAgentHarness(checkpoint_store=NullCheckpointStore(), tool_queue=queue)
        harness._enqueue_proposals(_request(), [_proposal()])
        tasks = queue.all()
        assert len(tasks) == 1
        assert tasks[0].tool_name == "write_conversation_message"
        assert tasks[0].priority == 1  # high -> 1
        assert tasks[0].payload["organization_id"] == 1
        assert tasks[0].payload["user_id"] == 2
    finally:
        cfg.config.enable_tool_queue = previous


def test_harness_does_not_enqueue_when_disabled() -> None:
    previous = cfg.config.enable_tool_queue
    cfg.config.enable_tool_queue = False
    try:
        harness = AllCallAllAgentHarness(checkpoint_store=NullCheckpointStore())
        assert harness._tool_queue is None
        harness._enqueue_proposals(_request(), [_proposal()])  # no-op
    finally:
        cfg.config.enable_tool_queue = previous


def test_worker_executes_and_acks() -> None:
    queue: AsyncToolQueue = AsyncToolQueue()
    queue.enqueue("t", {"x": 1}, idempotency_key="k")
    calls: list[str] = []

    def executor(task: object) -> None:
        calls.append(task.tool_name)  # type: ignore[attr-defined]

    worker = ToolQueueWorker(queue, executor, owner="w")
    assert worker.run_once() is True
    assert calls == ["t"]
    assert queue.all()[0].status == "done"


def test_worker_retries_then_dead_letters() -> None:
    queue: AsyncToolQueue = AsyncToolQueue()
    queue.enqueue("t", {}, idempotency_key="k", max_attempts=1)

    def executor(task: object) -> None:
        raise RuntimeError("boom")

    worker = ToolQueueWorker(queue, executor, owner="w")
    worker.run_once()
    assert queue.all()[0].status == "dead"


def test_tool_queue_status_endpoint_reports_counts() -> None:
    previous_token = cfg.config.api_token
    cfg.config.api_token = ""  # endpoint open for the test
    try:
        queue = get_default_tool_queue()
        queue.enqueue("status_tool", {}, idempotency_key="status-test-key-unique")
        client = TestClient(app)
        resp = client.get("/v1/tool-queue/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] == cfg.config.enable_tool_queue
        assert body["total"] >= 1
        assert body["queued"] >= 1
    finally:
        cfg.config.api_token = previous_token


def test_skills_endpoint_empty_when_disabled() -> None:
    previous_token = cfg.config.api_token
    previous_enabled = cfg.config.enable_skills
    previous_manifest = cfg.config.skill_manifest_path
    cfg.config.api_token = ""
    cfg.config.enable_skills = False
    cfg.config.skill_manifest_path = ""
    try:
        client = TestClient(app)
        resp = client.get("/v1/skills")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["skills"] == []
    finally:
        cfg.config.api_token = previous_token
        cfg.config.enable_skills = previous_enabled
        cfg.config.skill_manifest_path = previous_manifest


def test_skills_endpoint_resolves_with_security_overlay(tmp_path: Path) -> None:
    previous_token = cfg.config.api_token
    previous_enabled = cfg.config.enable_skills
    previous_manifest = cfg.config.skill_manifest_path
    cfg.config.api_token = ""
    cfg.config.enable_skills = True
    try:
        skill_md = tmp_path / "bulk_writer.md"
        skill_md.write_text(
            textwrap.dedent(
                """
                ---
                name: bulk_writer
                risk_level: low
                tools: [write_conversation_message]
                ---
                Write stuff.
                """
            ).strip(),
            encoding="utf-8",
        )
        manifest = tmp_path / "skills.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                skills:
                  - name: bulk_writer
                    path: bulk_writer.md
                    risk_level: high
                """
            ).strip(),
            encoding="utf-8",
        )
        cfg.config.skill_manifest_path = str(manifest)
        client = TestClient(app)
        resp = client.get("/v1/skills")
        assert resp.status_code == 200
        skills = resp.json()["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "bulk_writer"
        # Manifest floor (high) overrides the file's self-reported low -> overlay applied.
        assert skills[0]["risk_level"] == "high"
        assert skills[0]["requires_approval"] is True
    finally:
        cfg.config.api_token = previous_token
        cfg.config.enable_skills = previous_enabled
        cfg.config.skill_manifest_path = previous_manifest
