from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from allcallall_rag_runtime.config import config
from allcallall_rag_runtime.eval_runner import load_cases, run_eval
from allcallall_rag_runtime.main import app
from allcallall_rag_runtime.models import AgenticRetrievalRequest, ContextChunk
from allcallall_rag_runtime.qdrant_adapter import QdrantAdapter
from allcallall_rag_runtime.llamaindex_adapter import run_fixture_retrieval
from allcallall_rag_runtime.retrieval import (
    agentic_retrieve,
    build_graph_expansion,
    grounding_check,
    rerank,
    route_query,
)


def test_rules_rerank_prioritizes_relevant_source() -> None:
    chunks = [
        ContextChunk(
            chunk_id="msg",
            source_type="message",
            source_id="m1",
            snippet="approval appeared in a generic chat message",
            score=80,
        ),
        ContextChunk(
            chunk_id="kb",
            source_type="knowledge",
            source_id="k1",
            source_title="Approval policy",
            snippet="Supplier launch approval requires QA signoff and rollback review.",
            score=80,
        ),
    ]

    ranked = rerank("supplier launch approval policy", chunks, top_k=2).chunks

    assert ranked[0].source_type == "knowledge"
    assert ranked[0].final_rank == 1
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_agentic_retrieval_builds_evidence_pack() -> None:
    chunks = [
        ContextChunk(
            chunk_id="mt1",
            source_type="meeting_transcript",
            source_id="segment-1",
            snippet="The meeting identified supplier approval delay as the launch risk.",
            score=90,
        ),
        ContextChunk(
            chunk_id="mt2",
            source_type="meeting_transcript",
            source_id="segment-2",
            snippet="The meeting opened with general status updates.",
            score=10,
        )
    ]
    response = agentic_retrieve(
        AgenticRetrievalRequest(
            query="launch risk",
            source_types=["meeting_transcript"],
            chunks=chunks,
            top_k=1,
            min_confidence=0.6,
        ),
        chunks,
    )

    assert response.context_sufficiency.sufficient is True
    assert response.evidence_pack.selected_chunk_ids == ["mt1"]
    assert response.attempts[0].hit_count == 1
    assert response.route.intent == "risk"
    assert response.retrieval_route.intent == "risk"
    assert response.attempts[0].strategy == "multi_hop"
    assert response.raw_hits
    assert response.reranked_hits[0].chunk_id == "mt1"
    assert response.rejected_chunks[0].chunk_id == "mt2"


def test_dynamic_route_and_graph_expansion_for_consult() -> None:
    chunks = [
        ContextChunk(
            chunk_id="kb-launch",
            source_type="knowledge",
            source_id="doc-1",
            source_title="Launch readiness policy",
            snippet="Launch readiness requires QA signoff, owner approval, and rollback plan review.",
            score=90,
        )
    ]

    route = route_query("What does launch readiness require?", ["knowledge"], chunks)
    graph = build_graph_expansion("launch readiness require", chunks)
    response = agentic_retrieve(
        AgenticRetrievalRequest(
            query="What does launch readiness require?",
            source_types=["knowledge"],
            chunks=chunks,
            top_k=3,
            min_confidence=0.6,
        ),
        chunks,
    )

    assert route.intent == "consult"
    assert graph.enabled is True
    assert graph.edges[0].relation == "requires"
    assert response.graph_expansion.expanded_terms
    assert response.evidence_pack.route_intent == "consult"


def test_grounding_check_detects_missing_evidence() -> None:
    citation = ContextChunk(
        chunk_id="mt1",
        source_type="meeting_transcript",
        source_id="segment-1",
        snippet="Alice owns the supplier approval mitigation.",
    )

    grounded = grounding_check("Alice owns supplier approval mitigation", [citation])
    unsupported = grounding_check("Bob approved the budget increase", [citation])

    assert grounded.grounded is True
    assert unsupported.grounded is False
    assert unsupported.unsupported_claims


def test_eval_fixture_passes() -> None:
    fixture = Path(__file__).resolve().parents[1] / "evals" / "cases.json"

    report = run_eval(load_cases(fixture))

    assert report.summary.total_cases == 3
    assert report.summary.passed_cases == 3
    assert report.summary.grounding_pass_rate == 1


def test_metrics_endpoint_records_rerank_calls() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/retrieval/rerank",
        json={
            "query": "approval risk",
            "chunks": [
                {
                    "chunk_id": "mt1",
                    "source_type": "meeting_transcript",
                    "source_id": "1",
                    "snippet": "approval risk",
                }
            ],
        },
    )
    assert response.status_code == 200

    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "rag_runtime_rerank_total" in metrics.text


def test_qdrant_adapter_parses_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "vector_store", "qdrant")
    monkeypatch.setattr(config, "qdrant_url", "http://qdrant")
    monkeypatch.setattr(config, "qdrant_collection", "chunks")

    def fake_post(
        url: str,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        assert url == "http://qdrant/collections/chunks/points/search"
        assert json["vector"] == [0.1, 0.2]
        assert timeout == config.qdrant_timeout_sec
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": "point-1",
                        "score": 0.93,
                        "payload": {
                            "chunk_id": "qdrant-1",
                            "source_type": "knowledge",
                            "source_id": "doc-1",
                            "title": "Policy",
                            "snippet": "Qdrant vector retrieval supports payload-filtered chunks.",
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    chunks = QdrantAdapter().query(
        AgenticRetrievalRequest(query="policy", query_vector=[0.1, 0.2], top_k=3)
    )

    assert chunks[0].chunk_id == "qdrant-1"
    assert chunks[0].retrieval_mode == "qdrant_vector"


def test_llamaindex_eval_baseline_is_deterministic() -> None:
    result = run_fixture_retrieval(
        "approval checklist",
        [
            ContextChunk(source_type="message", source_id="1", snippet="general chat"),
            ContextChunk(source_type="knowledge", source_id="2", snippet="approval checklist requires QA signoff"),
        ],
        top_k=1,
    )

    assert result.hits[0].source_type == "knowledge"
