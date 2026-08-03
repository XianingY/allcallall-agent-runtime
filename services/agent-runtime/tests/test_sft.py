"""Tests for Part 2 SFT reflow: sample construction and JSONL export."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from allcallall_agent_runtime.badcase import BadcaseStore, classify_badcase
from allcallall_agent_runtime.models import (
    BadcaseCategory,
    BadcaseRecord,
    BadcaseSeverity,
    BadcaseSource,
    MeetingBriefRequest,
    MeetingBriefResponse,
    OutputDecision,
    ToolProposal,
)
from allcallall_agent_runtime.sft_dataset import (
    build_sft_dataset,
    build_sft_sample,
    export_sft_dataset,
)


def _req() -> MeetingBriefRequest:
    return MeetingBriefRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=99,
        goal="Summarize the Q3 risk review for the Alpha account.",
        context_chunks=[],
        notes=[],
    )


def _resp(**kw: object) -> MeetingBriefResponse:
    return MeetingBriefResponse(**kw)


def _labeled_record(category: BadcaseCategory, corrected: MeetingBriefResponse) -> BadcaseRecord:
    """Build a classified record, persist it, label it, and return the labeled row.

    Auto-detected categories go through ``classify_badcase``; human-labeled
    categories (e.g. UNSUPPORTED_MISHANDLE) are constructed directly with a
    SAMPLING_AUDIT source, since no runtime signal triggers them.
    """
    if category == BadcaseCategory.UNSUPPORTED_MISHANDLE:
        record = BadcaseRecord(
            id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            organization_id=1,
            workflow_run_id=99,
            source=BadcaseSource.SAMPLING_AUDIT,
            category=category,
            severity=BadcaseSeverity.LOW,
            request=_req(),
            response=_resp(status="ready", stop_reason="completed"),
            auto_signals={},
        )
    else:
        if category == BadcaseCategory.REVIEW_REJECT:
            raw = _resp(output_decision=OutputDecision(final_verdict="reject"))
        elif category == BadcaseCategory.APPROVAL_BYPASS:
            raw = _resp(proposed_tool_calls=[ToolProposal(tool_name="x", approval_required=False)])
        else:
            raw = _resp(status="failed", error="e", stop_reason="runtime_error")
        record = classify_badcase(_req(), raw)
        assert record is not None
    store = BadcaseStore(":memory:")
    store.save(record)
    assert store.label(record.id, corrected, by="alice")
    labeled = store.get(record.id)
    assert labeled is not None
    return labeled


def test_build_sft_sample_happy_path() -> None:
    corrected = _resp(
        status="ready",
        summary="The Alpha account Q3 risk is moderate; key driver is FX exposure.",
        citations=[],
    )
    record = _labeled_record(BadcaseCategory.REVIEW_REJECT, corrected)
    sample = build_sft_sample(record)
    assert sample is not None
    assert sample.badcase_id == record.id
    assert sample.messages[0].role == "user"
    assert "Summarize" in sample.messages[0].content
    assert sample.messages[1].role == "assistant"
    assert "moderate" in sample.messages[1].content
    assert "review_reject" in sample.tags
    assert sample.quality_score >= 0.6


def test_approval_bypass_sample_rejects_unsafe_correction() -> None:
    # Corrected response still contains a tool that skips approval -> invalid.
    corrected = _resp(
        summary="Applied the update.",
        proposed_tool_calls=[ToolProposal(tool_name="update", approval_required=False)],
    )
    record = _labeled_record(BadcaseCategory.APPROVAL_BYPASS, corrected)
    assert build_sft_sample(record) is None


def test_approval_bypass_sample_accepts_safe_correction() -> None:
    corrected = _resp(
        summary="Applied the update.",
        proposed_tool_calls=[ToolProposal(tool_name="update", approval_required=True)],
    )
    record = _labeled_record(BadcaseCategory.APPROVAL_BYPASS, corrected)
    sample = build_sft_sample(record)
    assert sample is not None
    assert "approval_bypass" in sample.tags


def test_unsupported_mishandle_sample_rejects_tool_proposals() -> None:
    corrected = _resp(
        status="ready",
        summary="Insufficient context to act.",
        proposed_tool_calls=[ToolProposal(tool_name="x", approval_required=True)],
    )
    record = _labeled_record(BadcaseCategory.UNSUPPORTED_MISHANDLE, corrected)
    assert build_sft_sample(record) is None


def test_not_eligible_returns_none() -> None:
    record = classify_badcase(_req(), _resp(status="failed", error="e", stop_reason="runtime_error"))
    assert record is not None
    # Not labeled -> sft_eligible is False -> no sample.
    assert record.sft_eligible is False
    assert build_sft_sample(record) is None


def test_missing_corrected_response_returns_none() -> None:
    # Label with an empty corrected response should still produce no sample.
    record = classify_badcase(_req(), _resp(status="failed", error="e", stop_reason="runtime_error"))
    assert record is not None
    store = BadcaseStore(":memory:")
    store.save(record)
    assert store.label(record.id, _resp(status="ready", summary=""), by="alice")
    labeled = store.get(record.id)
    assert labeled is not None
    assert build_sft_sample(labeled) is None


def test_export_sft_dataset_writes_jsonl(tmp_path: Path) -> None:
    corrected = _resp(status="ready", summary="The Alpha account Q3 risk is moderate.", citations=[])
    record = _labeled_record(BadcaseCategory.REVIEW_REJECT, corrected)
    out = tmp_path / "sft.jsonl"
    count = export_sft_dataset([record], out)
    assert count == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "messages" in lines[0]


def test_build_sft_dataset_dedups_by_badcase() -> None:
    corrected = _resp(status="ready", summary="ok", citations=[])
    record = _labeled_record(BadcaseCategory.REVIEW_REJECT, corrected)
    samples = build_sft_dataset([record, record])
    assert len(samples) == 1
