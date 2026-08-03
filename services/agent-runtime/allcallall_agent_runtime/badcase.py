"""Badcase capture: classify workflow failures and persist them for labeling/SFT.

Phase 1 scope (per design): automatic signals only -- ``AUTO_REVIEW`` (the
two-tier CheckAgent verdict) and ``AUTO_RUNTIME`` (failed status, timeout,
grounding failure, unsafe write proposal). Human-assigned categories
(HALLUCINATION / ROUTE_ERROR / USER_DECLINE / UNSUPPORTED_MISHANDLE) are filled
in during manual labeling (SAMPLING_AUDIT) or via the future user-feedback
signal (Phase 2). Storage reuses the standard-library sqlite3 single-connection
pattern already used by the checkpoint saver -- no new middleware.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import (
    BadcaseCategory,
    BadcaseRecord,
    BadcaseSeverity,
    BadcaseSource,
    WorkflowRequest,
    WorkflowResponse,
)

_SEVERITY_BY_CATEGORY: dict[BadcaseCategory, BadcaseSeverity] = {
    BadcaseCategory.APPROVAL_BYPASS: BadcaseSeverity.HIGH,
    BadcaseCategory.REVIEW_REJECT: BadcaseSeverity.HIGH,
    BadcaseCategory.HALLUCINATION: BadcaseSeverity.HIGH,
    BadcaseCategory.RETRIEVAL_MISS: BadcaseSeverity.MEDIUM,
    BadcaseCategory.ROUTE_ERROR: BadcaseSeverity.MEDIUM,
    BadcaseCategory.TIMEOUT: BadcaseSeverity.MEDIUM,
    BadcaseCategory.RUNTIME_ERROR: BadcaseSeverity.MEDIUM,
    BadcaseCategory.UNSUPPORTED_MISHANDLE: BadcaseSeverity.LOW,
    BadcaseCategory.USER_DECLINE: BadcaseSeverity.LOW,
}


def classify_badcase(
    request: WorkflowRequest,
    response: WorkflowResponse,
    *,
    source: BadcaseSource | None = None,
    now: datetime | None = None,
) -> BadcaseRecord | None:
    """Return a BadcaseRecord when ``response`` is a failure, else ``None``.

    Priority (most actionable first): unsafe write proposal > CheckAgent
    reject/escalate > genuine grounding failure > timeout > generic runtime
    failure. A clean (``ready``/``requires_action``, grounded, approved) response
    yields ``None`` so only real failures are archived.
    """
    signals: dict[str, Any] = {}
    verdict = response.output_decision.final_verdict if response.output_decision else "accept"
    failed = response.status == "failed"
    error = response.error or ""
    stop_reason = response.stop_reason or ""
    grounding_failed = stop_reason == "grounding_failed"
    approval_safe = all(item.approval_required for item in response.proposed_tool_calls)

    category: BadcaseCategory | None = None
    detected_source: BadcaseSource | None = None

    if not approval_safe:
        category = BadcaseCategory.APPROVAL_BYPASS
        detected_source = BadcaseSource.AUTO_REVIEW
        signals["unsafe_write_proposal"] = True
    elif verdict in ("reject", "escalate"):
        category = BadcaseCategory.REVIEW_REJECT
        detected_source = BadcaseSource.AUTO_REVIEW
        signals["verdict"] = verdict
    elif grounding_failed:
        category = BadcaseCategory.RETRIEVAL_MISS
        detected_source = BadcaseSource.AUTO_RUNTIME
        signals["grounding_failed"] = True
    elif "timeout" in error.lower() or "exceeded" in error.lower() or stop_reason == "timeout":
        category = BadcaseCategory.TIMEOUT
        detected_source = BadcaseSource.AUTO_RUNTIME
        signals["error"] = error[:200]
    elif failed or error or stop_reason == "runtime_error":
        category = BadcaseCategory.RUNTIME_ERROR
        detected_source = BadcaseSource.AUTO_RUNTIME
        signals["status"] = response.status
        signals["error"] = error[:200]
        signals["stop_reason"] = stop_reason

    if category is None:
        return None

    final_source = source or detected_source or BadcaseSource.AUTO_RUNTIME
    created_at = (now or datetime.now(timezone.utc)).isoformat()
    return BadcaseRecord(
        id=uuid.uuid4().hex,
        created_at=created_at,
        organization_id=request.organization_id,
        workflow_run_id=request.workflow_run_id,
        source=final_source,
        category=category,
        severity=_SEVERITY_BY_CATEGORY[category],
        request=request,
        response=response,
        auto_signals=signals,
    )


class BadcaseStore:
    """SQLite-backed store for BadcaseRecords (reuses the checkpoint saver idiom)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS badcases (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    organization_id INTEGER NOT NULL,
                    workflow_run_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sft_eligible INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def save(self, record: BadcaseRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO badcases "
                "(id, created_at, organization_id, workflow_run_id, source, category, "
                "severity, status, sft_eligible, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.created_at,
                    record.organization_id,
                    record.workflow_run_id,
                    record.source.value,
                    record.category.value,
                    record.severity.value,
                    record.status,
                    int(record.sft_eligible),
                    record.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get(self, record_id: str) -> BadcaseRecord | None:
        row = self._conn.execute("SELECT payload FROM badcases WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return None
        return BadcaseRecord.model_validate_json(row["payload"])

    def list_by_status(self, status: str, limit: int = 100) -> list[BadcaseRecord]:
        rows = self._conn.execute(
            "SELECT payload FROM badcases WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [BadcaseRecord.model_validate_json(r["payload"]) for r in rows]

    def list_open(self, limit: int = 100) -> list[BadcaseRecord]:
        return self.list_by_status("open", limit)

    def label(
        self,
        record_id: str,
        corrected_response: WorkflowResponse,
        *,
        note: str = "",
        by: str = "",
        sft_eligible: bool = True,
    ) -> bool:
        record = self.get(record_id)
        if record is None:
            return False
        record.label_corrected_response = corrected_response
        record.label_note = note
        record.labeled_by = by
        record.labeled_at = datetime.now(timezone.utc).isoformat()
        record.status = "labeled"
        record.sft_eligible = sft_eligible
        self.save(record)
        return True

    def set_sft_eligible(self, record_id: str, eligible: bool) -> bool:
        record = self.get(record_id)
        if record is None:
            return False
        record.sft_eligible = eligible
        if eligible and record.status == "open":
            record.status = "labeled"
        self.save(record)
        return True

    def list_sft_eligible(self, limit: int = 1000) -> list[BadcaseRecord]:
        rows = self._conn.execute(
            "SELECT payload FROM badcases WHERE sft_eligible=1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [BadcaseRecord.model_validate_json(r["payload"]) for r in rows]
