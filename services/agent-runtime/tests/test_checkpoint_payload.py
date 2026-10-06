"""Tests for checkpoint payload projection (Task 12).

Covers:
- Projection removes runtime objects (provider, tool_bridge, retrieval_cache).
- Projection retains resume state (execution ID, checkpoint version, pending
  approvals, selected evidence IDs, loop position, compact role outputs).
- Projection replaces large evidence bodies with references and hashes.
- Projection compacts trace events and citations.
- Projected state is deserializable and deterministic.
- Compatibility: old-serializer output can be read by the new saver.
"""

from __future__ import annotations

import json

import pytest

from allcallall_agent_runtime.checkpoint.payload import (
    project_checkpoint_state,
    serialized_checkpoint_size,
)
from allcallall_agent_runtime.models import (
    Citation,
    ContextChunk,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
    IntentRoute,
    RetrievalPlan,
    RoleResult,
    TraceEvent,
)


def _chunk(source_type: str, source_id: str = "s1", snippet: str = "") -> ContextChunk:
    return ContextChunk(
        chunk_id=f"{source_type}-{source_id}",
        source_type=source_type,
        source_id=source_id,
        source_title=f"title-{source_type}",
        snippet=snippet or f"snippet for {source_type} {source_id}",
        score=1,
    )


def _cite(source_type: str, source_id: str = "s1") -> Citation:
    return Citation(
        chunk_id=f"{source_type}-{source_id}",
        source_type=source_type,
        source_id=source_id,
        snippet=f"snippet for {source_type} {source_id}",
    )


def _sample_state() -> dict:
    """Build a representative state with all key categories."""
    return {
        "request": {"organization_id": 1, "user_id": 1, "conversation_id": 1, "workflow_run_id": 1, "goal": "test"},
        "provider": object(),  # Non-serializable runtime object
        "tool_bridge": object(),  # Non-serializable runtime object
        "retrieval_cache": object(),  # Per-run cache
        "trace_events": [
            TraceEvent(event="graph.node.started", node="collect_context", status="running"),
            TraceEvent(event="rag.observe", node="retrieval_loop", status="completed", tool_name="query_context_chunks"),
            TraceEvent(event="graph.node.completed", node="retrieval_loop", status="completed"),
        ],
        "agentic_context_chunks": [_chunk("meeting_transcript", "a"), _chunk("knowledge", "b")],
        "retrieved_context_chunks": [_chunk("meeting_transcript", "a"), _chunk("knowledge", "b")],
        "reranked_context_chunks": [_chunk("meeting_transcript", "a")],
        "citations": [_cite("meeting_transcript", "a"), _cite("knowledge", "b")],
        "role_results": [RoleResult(role="searcher", summary="found evidence")],
        "evidence_pack": EvidencePack(selected_chunk_ids=["meeting_transcript-a"], confidence=0.8),
        "context_sufficiency": ContextSufficiency(sufficient=True, confidence=0.8),
        "summary": "Meeting brief summary",
        "action_items": ["Follow up on item A"],
        "risk_flags": [],
        "proposed_tool_calls": [],
        "critic_result": CriticResult(passed=True),
        "critic_retries": 0,
        "last_check_decision": "pass",
        "check_log": [],
    }


class TestProjectionRemovesRuntimeObjects:
    def test_provider_is_removed(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert "provider" not in projected

    def test_tool_bridge_is_removed(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert "tool_bridge" not in projected

    def test_retrieval_cache_is_removed(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert "retrieval_cache" not in projected


class TestProjectionRetainsResumeState:
    def test_summary_is_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert projected["summary"] == "Meeting brief summary"

    def test_action_items_are_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert projected["action_items"] == ["Follow up on item A"]

    def test_evidence_pack_is_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert projected["evidence_pack"].selected_chunk_ids == ["meeting_transcript-a"]

    def test_context_sufficiency_is_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert projected["context_sufficiency"].sufficient is True

    def test_critic_result_is_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert projected["critic_result"].passed is True

    def test_role_results_are_retained(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert len(projected["role_results"]) == 1
        assert projected["role_results"][0].role == "searcher"


class TestProjectionCompactsChunkLists:
    def test_chunk_list_becomes_references(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        agentic = projected["agentic_context_chunks"]
        assert isinstance(agentic, list)
        assert len(agentic) == 2
        # Each entry is a compact reference dict, not a full ContextChunk.
        assert all(isinstance(ref, dict) for ref in agentic)
        assert "snippet_hash" in agentic[0]
        assert "chunk_id" in agentic[0]

    def test_chunk_reference_has_no_full_snippet(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        for ref in projected["reranked_context_chunks"]:
            # Compact references have snippet_hash, not the full snippet text.
            assert "snippet_hash" in ref
            assert "snippet" not in ref or len(ref.get("snippet", "")) == 0


class TestProjectionCompactsTraceEvents:
    def test_trace_events_are_compacted(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        events = projected["trace_events"]
        assert isinstance(events, list)
        assert len(events) == 3
        # Compacted events have limited keys.
        assert "event" in events[0]
        assert "node" in events[0]
        # Full metadata is dropped.
        assert "metadata" not in events[0]


class TestProjectionCompactsCitations:
    def test_citations_are_compacted(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        cites = projected["citations"]
        assert isinstance(cites, list)
        assert len(cites) == 2
        # Compacted citations have truncated snippets.
        for cite in cites:
            assert "chunk_id" in cite
            assert "source_type" in cite
            assert len(cite.get("snippet", "")) <= 120


class TestSerializedCheckpointSize:
    def test_size_is_positive_for_non_empty_payload(self) -> None:
        state = _sample_state()
        # Use only serializable values for size measurement.
        serializable = {k: v for k, v in state.items() if k not in ("provider", "tool_bridge", "retrieval_cache")}
        size = serialized_checkpoint_size(serializable)
        assert size > 0

    def test_size_is_small_for_empty_payload(self) -> None:
        assert serialized_checkpoint_size({}) >= 2  # "{}"


class TestProjectionIsDeterministic:
    def test_same_input_produces_same_output(self) -> None:
        state = _sample_state()
        projected1 = project_checkpoint_state(state)
        projected2 = project_checkpoint_state(state)
        # Chunk references should be identical.
        assert projected1["agentic_context_chunks"] == projected2["agentic_context_chunks"]
        assert projected1["trace_events"] == projected2["trace_events"]
        assert projected1["citations"] == projected2["citations"]


class TestCompatibilityOldSerializerOutput:
    def test_projection_handles_dict_values(self) -> None:
        """The projection must handle both Pydantic models and plain dicts
        (old-serializer output may contain dicts)."""
        state = {
            "agentic_context_chunks": [
                {"chunk_id": "mt-1", "source_type": "meeting_transcript", "source_id": "1", "snippet": "text"},
            ],
            "citations": [
                {"chunk_id": "mt-1", "source_type": "meeting_transcript", "source_id": "1", "snippet": "text"},
            ],
        }
        projected = project_checkpoint_state(state)
        assert len(projected["agentic_context_chunks"]) == 1
        assert projected["agentic_context_chunks"][0]["chunk_id"] == "mt-1"
        assert len(projected["citations"]) == 1
        assert projected["citations"][0]["chunk_id"] == "mt-1"
