"""SFT reflow: convert labeled BadcaseRecords into supervised fine-tuning samples.

Part 2 of the badcase -> SFT -> online-eval loop. Produces standard
``messages``-format chat samples (one per labeled, SFT-eligible badcase) that
can be exported as JSONL for an external fine-tuning platform. The provider is
still a ``rules`` placeholder with no trainable weights of its own, so this
module only constructs and exports the dataset -- actual training happens
offline on the dataset we emit.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    BadcaseCategory,
    BadcaseRecord,
    ChatMessage,
    SFTSample,
    WorkflowResponse,
)

# Samples below this quality score are dropped to keep the training set clean.
_QUALITY_THRESHOLD = 0.6


def _user_content(badcase: BadcaseRecord) -> str:
    request = badcase.request
    parts: list[str] = [f"Goal: {request.goal}"]
    if request.context_chunks:
        chunks = "\n".join(f"- {chunk.snippet}" for chunk in request.context_chunks[:8])
        parts.append(f"Context:\n{chunks}")
    if request.notes:
        notes = "\n".join(f"- {note.body}" for note in request.notes[:5])
        parts.append(f"Notes:\n{notes}")
    return "\n\n".join(parts)


def _assistant_content(corrected: WorkflowResponse, category: BadcaseCategory) -> str | None:
    """Format the corrected response as the assistant turn.

    Returns ``None`` when the corrected response violates a category-specific
    invariant (e.g. an approval-bypass sample whose corrected tools still skip
    approval), because such a sample would teach the wrong behavior.
    """
    if category == BadcaseCategory.APPROVAL_BYPASS:
        if not all(item.approval_required for item in corrected.proposed_tool_calls):
            return None
    if category == BadcaseCategory.UNSUPPORTED_MISHANDLE:
        if corrected.proposed_tool_calls:
            return None
    lines: list[str] = []
    if corrected.summary:
        lines.append(corrected.summary)
    if corrected.citations:
        sources = "\n".join(
            f"- {c.title or c.source_title} ({c.source_type})" for c in corrected.citations
        )
        lines.append(f"Sources:\n{sources}")
    if corrected.proposed_tool_calls:
        proposals = "\n".join(
            f"- {p.tool_name}: {p.reason}" for p in corrected.proposed_tool_calls
        )
        lines.append(f"Proposed actions:\n{proposals}")
    return "\n\n".join(lines) if lines else None


def _quality_score(corrected: WorkflowResponse, category: BadcaseCategory) -> float:
    score = 0.5
    if corrected.summary:
        score += 0.2
    if corrected.citations:
        score += 0.2
    if category == BadcaseCategory.UNSUPPORTED_MISHANDLE and not corrected.proposed_tool_calls:
        score += 0.1
    if category == BadcaseCategory.APPROVAL_BYPASS and all(
        item.approval_required for item in corrected.proposed_tool_calls
    ):
        score += 0.1
    return min(1.0, score)


def build_sft_sample(badcase: BadcaseRecord) -> SFTSample | None:
    """Build an SFT sample from a labeled, SFT-eligible badcase.

    Returns ``None`` when the record is not eligible/labeled, lacks a corrected
    response, the corrected response violates a category invariant, or the
    resulting sample scores below the quality threshold.
    """
    if not badcase.sft_eligible or badcase.status not in ("labeled", "approved", "training"):
        return None
    corrected = badcase.label_corrected_response
    if corrected is None:
        return None
    user_text = _user_content(badcase)
    assistant_text = _assistant_content(corrected, badcase.category)
    if not user_text or not assistant_text:
        return None
    quality_score = _quality_score(corrected, badcase.category)
    if quality_score < _QUALITY_THRESHOLD:
        return None
    return SFTSample(
        id=uuid.uuid4().hex,
        badcase_id=badcase.id,
        model_version=badcase.response.prompt_version or "unknown",
        route=badcase.response.route_decision.route,
        system_prompt=(
            f"[AllCallAll agent-runtime] prompt_version={badcase.response.prompt_version} "
            f"route={badcase.response.route_decision.route}"
        ),
        messages=[
            ChatMessage(role="user", content=user_text),
            ChatMessage(role="assistant", content=assistant_text),
        ],
        tags=[badcase.category.value, badcase.severity.value, badcase.source.value],
        quality_score=quality_score,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_sft_dataset(records: list[BadcaseRecord]) -> list[SFTSample]:
    """Build de-duplicated SFT samples from a batch of badcase records."""
    samples: list[SFTSample] = []
    seen: set[str] = set()
    for record in records:
        sample = build_sft_sample(record)
        if sample is None:
            continue
        if sample.badcase_id in seen:
            continue
        seen.add(sample.badcase_id)
        samples.append(sample)
    return samples


def export_sft_dataset(records: list[BadcaseRecord], out_path: Path) -> int:
    """Write eligible samples as JSONL. Returns the number of samples written."""
    samples = build_sft_dataset(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample.model_dump_json() + "\n")
    return len(samples)
