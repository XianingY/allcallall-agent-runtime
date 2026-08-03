"""Tests for Part 1 badcase capture: classifier, store, and harness wiring."""

from __future__ import annotations

import pytest

from allcallall_agent_runtime.badcase import BadcaseStore, classify_badcase
from allcallall_agent_runtime.harness import AllCallAllAgentHarness
from allcallall_agent_runtime.models import (
    BadcaseCategory,
    BadcaseSource,
    MeetingBriefRequest,
    MeetingBriefResponse,
    OutputDecision,
    ToolProposal,
)
from allcallall_agent_runtime.tool_layer import StubToolLayer


def _req() -> MeetingBriefRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=99,
        goal="g",
    )


def _resp(**kw: object) -> MeetingBriefResponse:
    return MeetingBriefResponse(**kw)


def test_classify_review_reject() -> None:
    resp = _resp(output_decision=OutputDecision(final_verdict="reject"))
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.REVIEW_REJECT
    assert rec.source == BadcaseSource.AUTO_REVIEW
    assert rec.severity.value == "high"


def test_classify_escalate_is_review_reject() -> None:
    resp = _resp(output_decision=OutputDecision(final_verdict="escalate"))
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.REVIEW_REJECT
    assert rec.auto_signals["verdict"] == "escalate"


def test_classify_approval_bypass() -> None:
    resp = _resp(proposed_tool_calls=[ToolProposal(tool_name="x", approval_required=False)])
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.APPROVAL_BYPASS
    assert rec.severity.value == "high"


def test_classify_grounding_failure() -> None:
    resp = _resp(status="ready", stop_reason="grounding_failed")
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.RETRIEVAL_MISS
    assert rec.source == BadcaseSource.AUTO_RUNTIME


def test_classify_timeout() -> None:
    resp = _resp(
        status="failed",
        error="workflow run exceeded the 120s request timeout",
        stop_reason="runtime_error",
    )
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.TIMEOUT


def test_classify_runtime_error() -> None:
    resp = _resp(status="failed", error="boom", stop_reason="runtime_error")
    rec = classify_badcase(_req(), resp)
    assert rec is not None
    assert rec.category == BadcaseCategory.RUNTIME_ERROR


def test_classify_clean_returns_none() -> None:
    resp = _resp(status="ready", stop_reason="completed")
    assert classify_badcase(_req(), resp) is None


def test_store_save_get_and_dedup() -> None:
    store = BadcaseStore(":memory:")
    rec = classify_badcase(_req(), _resp(status="failed", error="e", stop_reason="runtime_error"))
    assert rec is not None
    store.save(rec)
    got = store.get(rec.id)
    assert got is not None
    assert got.id == rec.id
    # INSERT OR REPLACE dedups by id
    store.save(rec)
    assert len(store.list_open()) == 1


def test_store_label_and_sft_eligible() -> None:
    store = BadcaseStore(":memory:")
    rec = classify_badcase(_req(), _resp(status="failed", error="e", stop_reason="runtime_error"))
    assert rec is not None
    store.save(rec)
    assert store.label(rec.id, _resp(status="ready"), note="fixed", by="alice")
    labeled = store.get(rec.id)
    assert labeled is not None
    assert labeled.status == "labeled"
    assert labeled.sft_eligible is True
    assert labeled.labeled_by == "alice"
    assert len(store.list_sft_eligible()) == 1


def test_store_set_sft_eligible_and_reopen_status() -> None:
    store = BadcaseStore(":memory:")
    rec = classify_badcase(_req(), _resp(status="failed", error="e", stop_reason="runtime_error"))
    assert rec is not None
    store.save(rec)
    assert store.set_sft_eligible(rec.id, True)
    labeled = store.get(rec.id)
    assert labeled is not None
    assert labeled.status == "labeled"
    assert store.set_sft_eligible(rec.id, False)
    reopened = store.get(rec.id)
    assert reopened is not None
    assert reopened.sft_eligible is False
    assert len(store.list_sft_eligible()) == 0


def test_harness_capture_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    store = BadcaseStore(":memory:")
    harness = AllCallAllAgentHarness(badcase_store=store, tool_layer=StubToolLayer())
    monkeypatch.setattr(
        "allcallall_agent_runtime.harness.app_config.enable_badcase_capture", True
    )
    resp = harness._failure_response(_req(), "rules", "boom", trace=[])
    assert resp.status == "failed"
    records = store.list_open()
    assert len(records) == 1
    assert records[0].category == BadcaseCategory.RUNTIME_ERROR


def test_harness_capture_disabled_no_store(monkeypatch: pytest.MonkeyPatch) -> None:
    store = BadcaseStore(":memory:")
    harness = AllCallAllAgentHarness(badcase_store=store, tool_layer=StubToolLayer())
    monkeypatch.setattr(
        "allcallall_agent_runtime.harness.app_config.enable_badcase_capture", False
    )
    harness._failure_response(_req(), "rules", "boom", trace=[])
    assert store.list_open() == []
