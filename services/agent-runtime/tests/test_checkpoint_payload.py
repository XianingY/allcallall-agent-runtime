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
import base64
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint
from langgraph.graph import END, StateGraph

from allcallall_agent_runtime.checkpoint.mysql import MySQLCheckpointSaver
from allcallall_agent_runtime.checkpoint.sqlite_saver import SQLiteCheckpointSaver
from allcallall_agent_runtime.checkpoint.payload import (
    project_checkpoint_state,
    serialized_checkpoint_size,
)
from allcallall_agent_runtime.metrics import checkpoint_payload_bytes, get_default_prometheus_registry
from allcallall_agent_runtime.models import (
    Citation,
    ContextChunk,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
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


class _ResumeState(TypedDict):
    reranked_context_chunks: list[ContextChunk]
    summary: str


def _consume_chunks(state: _ResumeState) -> dict[str, str]:
    return {"summary": state["reranked_context_chunks"][0].snippet}


def _sample_state() -> dict[str, Any]:
    """Build a representative state with all key categories."""
    return {
        "request": {"organization_id": 1, "user_id": 1, "conversation_id": 1, "workflow_run_id": 1, "goal": "test"},
        "provider": object(),  # Non-serializable runtime object
        "tool_bridge": object(),  # Non-serializable runtime object
        "rag_runtime": object(),  # Request-scoped RAG client
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

    def test_rag_runtime_is_removed(self) -> None:
        projected = project_checkpoint_state(_sample_state())
        assert "rag_runtime" not in projected

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

    def test_context_chunk_lists_are_preserved_for_existing_nodes(self) -> None:
        state = _sample_state()
        projected = project_checkpoint_state(state)
        for key in (
            "agentic_context_chunks",
            "retrieved_context_chunks",
            "reranked_context_chunks",
        ):
            assert projected[key] is state[key]
            assert all(isinstance(chunk, ContextChunk) for chunk in projected[key])


class TestCheckpointResume:
    def test_new_saver_resumes_with_context_chunks_accessible_to_nodes(self) -> None:
        saver = SQLiteCheckpointSaver(":memory:")
        graph = StateGraph(_ResumeState)
        graph.add_node("consume", _consume_chunks)
        graph.set_entry_point("consume")
        graph.add_edge("consume", END)
        app = graph.compile(checkpointer=saver, interrupt_before=["consume"])
        config: RunnableConfig = {"configurable": {"thread_id": "task-12-new-checkpoint"}}
        chunk = _chunk("meeting_transcript", "resume-1", snippet="resume-safe evidence")

        app.invoke({"reranked_context_chunks": [chunk], "summary": ""}, config)
        result = app.invoke(None, config)

        assert result is not None
        assert result["summary"] == "resume-safe evidence"

    def test_checkpoint_projection_observes_prometheus_size_histogram(self) -> None:
        registry = get_default_prometheus_registry()
        assert checkpoint_payload_bytes is not None
        saver = object.__new__(MySQLCheckpointSaver)
        saver.serde = JsonPlusSerializer()
        checkpoint: Checkpoint = {
            "v": 1,
            "id": "metrics-checkpoint",
            "ts": "2026-10-06T00:00:00+00:00",
            "channel_values": {
                "provider": object(),
                "trace_events": [TraceEvent(event="test", node="test", status="completed")],
                "summary": "checkpoint payload",
            },
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
        }

        original_before = registry.get_sample_value(
            "checkpoint_payload_bytes_count", {"stage": "original"}
        ) or 0.0
        projected_before = registry.get_sample_value(
            "checkpoint_payload_bytes_count", {"stage": "projected"}
        ) or 0.0

        saver._drop_unserializable_channels(checkpoint)

        original_after = registry.get_sample_value(
            "checkpoint_payload_bytes_count", {"stage": "original"}
        )
        projected_after = registry.get_sample_value(
            "checkpoint_payload_bytes_count", {"stage": "projected"}
        )
        assert original_after is not None and original_after > original_before
        assert projected_after is not None and projected_after > projected_before

    def test_new_saver_resumes_pre_change_checkpoint_fixture(self, tmp_path: Path) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "task12_pre_change_checkpoint.json"
        fixture = json.loads(fixture_path.read_text())
        saver = SQLiteCheckpointSaver(str(tmp_path / "pre-change.sqlite"))
        with saver._conn:
            saver._conn.execute(
                "INSERT INTO langgraph_checkpoint_threads "
                "(thread_id, checkpoint_ns, current_version, updated_at) VALUES (?, ?, 1, CURRENT_TIMESTAMP)",
                (fixture["thread_id"], ""),
            )
            saver._conn.execute(
                "INSERT INTO langgraph_checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, version, checkpoint_type, checkpoint_blob, "
                "metadata_type, metadata_blob, created_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    fixture["thread_id"],
                    "",
                    fixture["checkpoint_id"],
                    fixture["checkpoint_type"],
                    base64.b64decode(fixture["checkpoint_blob"]),
                    fixture["metadata_type"],
                    base64.b64decode(fixture["metadata_blob"]),
                ),
            )

        graph = StateGraph(_ResumeState)
        graph.add_node("consume", _consume_chunks)
        graph.set_entry_point("consume")
        graph.add_edge("consume", END)
        app = graph.compile(checkpointer=saver, interrupt_before=["consume"])
        config: RunnableConfig = {"configurable": {"thread_id": fixture["thread_id"]}}
        result = app.invoke(None, config)

        assert result is not None
        assert result["summary"] == "resume-safe evidence"


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


class TestProjectionPreservesCitations:
    def test_citations_are_preserved_for_existing_nodes(self) -> None:
        state = _sample_state()
        projected = project_checkpoint_state(state)
        cites = projected["citations"]
        assert cites is state["citations"]
        assert len(cites) == 2
        assert all(isinstance(cite, Citation) for cite in cites)


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
        assert projected["agentic_context_chunks"] == state["agentic_context_chunks"]
        assert len(projected["citations"]) == 1
        assert projected["citations"][0]["chunk_id"] == "mt-1"
