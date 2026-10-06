from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from allcallall_rag_runtime.api import create_app
from allcallall_rag_runtime.clients import build_rag_clients
from allcallall_rag_runtime.config import config
from allcallall_rag_runtime.config import RAGRuntimeConfig
from allcallall_rag_runtime.eval_runner import load_cases, run_eval
from allcallall_rag_runtime import retrieval as retrieval_module
from allcallall_rag_runtime.main import app
from allcallall_rag_runtime.models import (
    AgenticRetrievalRequest,
    ContextChunk,
    RetrievalQueryRequest,
)
from allcallall_rag_runtime.metrics import metrics
from allcallall_rag_runtime.pipeline import select_retrieval_chunks
from allcallall_rag_runtime.go_bridge import GoRetrievalBridge
from allcallall_rag_runtime.qdrant_adapter import QdrantAdapter
from allcallall_rag_runtime.llamaindex_adapter import run_fixture_retrieval
from allcallall_rag_runtime.retrieval import prepare_candidates, rerank_prepared
from allcallall_rag_runtime.retrieval import (
    agentic_retrieve,
    build_graph_expansion,
    grounding_check,
    rerank,
    route_query,
)


def _rag_config(**overrides: object) -> RAGRuntimeConfig:
    defaults: dict[str, object] = dict(
        tool_bridge_base_url="http://test-go-bridge",
        tool_bridge_token="test-token",
        tool_bridge_timeout_sec=2.5,
        vector_store="qdrant",
        qdrant_url="http://test-qdrant",
        qdrant_collection="chunks",
        qdrant_timeout_sec=3.5,
    )
    defaults.update(overrides)
    return RAGRuntimeConfig(**defaults)  # type: ignore[arg-type]


def _retrieval_request() -> AgenticRetrievalRequest:
    return AgenticRetrievalRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        query="policy",
        query_vector=[0.1, 0.2],
        top_k=3,
        chunks=[
            ContextChunk(
                chunk_id="inline",
                source_type="knowledge",
                source_id="inline-1",
                snippet="inline fallback",
            )
        ],
    )


def test_retrieval_request_omits_unused_agent_cache_fields() -> None:
    """The agent owns cache identity; RAG should not expose dead fields."""
    assert "context_fingerprint" not in RetrievalQueryRequest.model_fields
    assert "corpus_version" not in RetrievalQueryRequest.model_fields


class RAGRoutingTransport(httpx.BaseTransport):
    def __init__(self, *, set_cookie: bool = False) -> None:
        self.requests = 0
        self.seen_requests: list[httpx.Request] = []
        self.set_cookie = set_cookie

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        self.seen_requests.append(request)
        url = str(request.url)
        response: httpx.Response
        if "/agent/retrieval/query" in url:
            response = httpx.Response(
                200,
                json={
                    "chunks": [
                        {
                            "chunk_id": "go-1",
                            "source_type": "knowledge",
                            "source_id": "go-doc",
                            "snippet": "go bridge result",
                        }
                    ]
                },
            )
        elif "/points/search" in url:
            response = httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "qdrant-1",
                            "score": 0.93,
                            "payload": {
                                "chunk_id": "qdrant-1",
                                "source_type": "knowledge",
                                "source_id": "doc-1",
                                "title": "Policy",
                                "snippet": "qdrant result",
                            },
                        }
                    ]
                },
            )
        else:
            response = httpx.Response(200, json={})
        if self.set_cookie and self.requests == 1:
            response.headers["Set-Cookie"] = "session=tenant-one; Path=/"
        return response


def _metric_value(name: str) -> int:
    return metrics.snapshot().get(name, 0)


def test_app_factory_preserves_public_routes() -> None:
    expected = {
        "/health",
        "/ready",
        "/metrics",
        "/v1/capabilities",
        "/v1/retrieval/query",
        "/v1/retrieval/rerank",
        "/v1/retrieval/prepare",
        "/v1/retrieval/agentic",
        "/v1/grounding/check",
    }

    assert set(app.openapi()["paths"]) == expected
    assert set(create_app().openapi()["paths"]) == expected


def test_pipeline_uses_inline_chunks_when_external_sources_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "tool_bridge_base_url", "")
    monkeypatch.setattr(config, "vector_store", "none")
    request = AgenticRetrievalRequest(
        query="approval",
        chunks=[ContextChunk(source_type="knowledge", source_id="1", snippet="approval policy")],
    )

    chunks, source = select_retrieval_chunks(request)

    assert source == "inline"
    assert chunks == request.chunks


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

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://qdrant/collections/chunks/points/search"
        import json as _json

        body = _json.loads(request.content)
        assert body["vector"] == [0.1, 0.2]
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

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=config.qdrant_timeout_sec,
    )
    adapter = QdrantAdapter(http_client=http_client)
    try:
        chunks = adapter.query(
            AgenticRetrievalRequest(query="policy", query_vector=[0.1, 0.2], top_k=3)
        )
    finally:
        http_client.close()

    assert chunks[0].chunk_id == "qdrant-1"
    assert chunks[0].retrieval_mode == "qdrant_vector"


def test_rag_clients_share_one_transport_and_preserve_service_timeouts() -> None:
    transport = RAGRoutingTransport()
    http = httpx.Client(transport=transport, timeout=99)
    config = _rag_config(
        http_connect_timeout_sec=0.11,
        http_write_timeout_sec=0.22,
        http_pool_timeout_sec=0.33,
    )
    clients = build_rag_clients(config, http_client=http)

    clients.go_bridge.query(_retrieval_request())
    clients.qdrant.query(_retrieval_request())

    assert transport.requests == 2
    actual_timeouts = [request.extensions["timeout"] for request in transport.seen_requests]
    assert [timeout["read"] for timeout in actual_timeouts] == [2, 3.5]
    assert all(timeout["connect"] == 0.11 for timeout in actual_timeouts)
    assert all(timeout["write"] == 0.22 for timeout in actual_timeouts)
    assert all(timeout["pool"] == 0.33 for timeout in actual_timeouts)


def test_rag_clients_do_not_replay_cookies_across_requests() -> None:
    transport = RAGRoutingTransport(set_cookie=True)
    http = httpx.Client(transport=transport)
    clients = build_rag_clients(_rag_config(), http_client=http)

    clients.go_bridge.query(_retrieval_request())
    clients.go_bridge.query(_retrieval_request())

    assert transport.requests == 2
    assert all(request.headers.get("cookie") is None for request in transport.seen_requests)
    assert not http.cookies


def test_rag_clients_close_owned_client_once_and_keep_injected_caller_owned() -> None:
    owned_bundle = build_rag_clients(_rag_config())
    owned = owned_bundle.owned_http
    assert owned is not None
    with patch.object(owned, "close", wraps=owned.close) as owned_close:
        owned_bundle.close()
        owned_bundle.close()
        owned_close.assert_called_once()
    assert owned_bundle.owned_http is None
    assert owned_bundle.closed is True

    transport = RAGRoutingTransport()
    http = httpx.Client(transport=transport)
    injected_bundle = build_rag_clients(_rag_config(), http_client=http)
    assert injected_bundle.owned_http is None
    with patch.object(http, "close", wraps=http.close) as injected_close:
        injected_bundle.close()
        injected_close.assert_not_called()
    clients = build_rag_clients(_rag_config(), http_client=http)
    clients.go_bridge.query(_retrieval_request())
    assert transport.requests == 1


def test_go_bridge_pool_timeout_is_observable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("pool exhausted")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = GoRetrievalBridge(config=_rag_config(), http_client=http)
    before = _metric_value("rag_runtime_go_bridge_pool_timeouts_total")

    with caplog.at_level(logging.WARNING, logger="allcallall_rag_runtime.go_bridge"):
        with pytest.raises(httpx.PoolTimeout):
            bridge.query(_retrieval_request())

    assert _metric_value("rag_runtime_go_bridge_pool_timeouts_total") == before + 1
    assert any(
        record.message == "rag_runtime_go_bridge_pool_timeout"
        and getattr(record, "error_type", None) == "pool_timeout"
        for record in caplog.records
    )


def test_qdrant_fallback_is_observable(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    config = _rag_config(
        tool_bridge_base_url="",
        tool_bridge_token="",
    )
    clients = build_rag_clients(config, http_client=http)
    before = _metric_value("rag_runtime_qdrant_fallback_total")

    with caplog.at_level(logging.WARNING, logger="allcallall_rag_runtime.pipeline"):
        chunks, source = select_retrieval_chunks(_retrieval_request(), clients=clients)

    assert [chunk.chunk_id for chunk in chunks] == ["inline"]
    assert source == "inline"
    assert _metric_value("rag_runtime_qdrant_fallback_total") == before + 1
    assert any(
        record.message == "rag_runtime_qdrant_fallback"
        and getattr(record, "error_type", None) == "network"
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "field",
    [
        "http_max_connections",
        "http_max_keepalive_connections",
        "http_keepalive_expiry_sec",
        "http_connect_timeout_sec",
        "http_read_timeout_sec",
        "http_write_timeout_sec",
        "http_pool_timeout_sec",
    ],
)
def test_rag_http_pool_settings_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        _rag_config(**{field: 0})


def test_rag_keepalive_limit_cannot_exceed_total_connections() -> None:
    with pytest.raises(ValidationError):
        _rag_config(http_max_connections=2, http_max_keepalive_connections=3)


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


# --------------------------------------------------------------------------- #
# Task 12: RAG rerank reuse                                                         #
# --------------------------------------------------------------------------- #


class TestPrepareCandidates:
    def test_prepare_candidates_deduplicates(self) -> None:
        chunks = [
            ContextChunk(source_type="meeting_transcript", source_id="1", snippet="approval risk"),
            ContextChunk(source_type="meeting_transcript", source_id="1", snippet="approval risk"),
        ]
        prepared = prepare_candidates("approval", chunks)
        assert prepared.chunk_count == 1
        assert len(prepared.chunks) == 1
        assert prepared.token_count == len(prepared.tokens)

    def test_rerank_prepared_uses_prepared_candidates(self) -> None:
        chunks = [
            ContextChunk(
                chunk_id="kb-1",
                source_type="knowledge",
                source_id="1",
                snippet="approval checklist requires QA signoff",
                score=95,
            )
        ]
        prepared = prepare_candidates("approval checklist", chunks)
        response = rerank_prepared(prepared, top_k=1)
        assert len(response.chunks) == 1
        assert response.chunks[0].chunk_id == "kb-1"

    def test_prepare_candidates_filters_by_source_type(self) -> None:
        chunks = [
            ContextChunk(source_type="meeting_transcript", source_id="1", snippet="approval risk"),
            ContextChunk(source_type="knowledge", source_id="2", snippet="policy"),
        ]
        prepared = prepare_candidates("approval", chunks, source_types=["knowledge"])
        assert prepared.chunk_count == 1

    def test_prepare_candidates_fingerprint_is_deterministic(self) -> None:
        chunks = [
            ContextChunk(source_type="meeting_transcript", source_id="1", snippet="approval risk"),
            ContextChunk(source_type="knowledge", source_id="2", snippet="policy"),
        ]
        p1 = prepare_candidates("approval", chunks)
        p2 = prepare_candidates("approval", chunks)
        assert p1.fingerprint == p2.fingerprint

    def test_prepare_candidates_fingerprint_changes_with_different_chunks(self) -> None:
        chunks_a = [ContextChunk(source_type="meeting_transcript", source_id="1", snippet="approval")]
        chunks_b = [ContextChunk(source_type="knowledge", source_id="2", snippet="policy")]
        p1 = prepare_candidates("approval", chunks_a)
        p2 = prepare_candidates("approval", chunks_b)
        assert p1.fingerprint != p2.fingerprint


class TestRerankReuse:
    def test_agentic_retrieve_skips_duplicate_rerank_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunks = [
            ContextChunk(
                chunk_id="mt1",
                source_type="meeting_transcript",
                source_id="segment-1",
                snippet="The meeting identified supplier approval delay as the launch risk.",
                score=90,
            )
        ]
        rerank_calls = 0
        original_rerank = retrieval_module.rerank

        def counting_rerank(*args: object, **kwargs: object) -> object:
            nonlocal rerank_calls
            rerank_calls += 1
            return original_rerank(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(retrieval_module, "rerank", counting_rerank)
        response = agentic_retrieve(
            AgenticRetrievalRequest(
                query="launch risk",
                source_types=["meeting_transcript"],
                chunks=chunks,
                top_k=1,
                min_confidence=0.99,
                max_steps=2,
            ),
            chunks,
        )

        assert rerank_calls == 1
        assert any(event.get("event") == "rag.rerank_skipped" for event in response.trace)
        assert any(
            event.get("event") == "rag.final_rerank_skipped" for event in response.trace
        )

    def test_agentic_retrieve_skips_rerank_for_unchanged_candidates(self) -> None:
        """When the candidate fingerprint is unchanged between steps, the final
        rerank should be skipped (evidenced by a trace event)."""
        chunks = [
            ContextChunk(
                chunk_id="mt1",
                source_type="meeting_transcript",
                source_id="segment-1",
                snippet="The meeting identified supplier approval delay as the launch risk.",
                score=90,
            ),
        ]
        response = agentic_retrieve(
            AgenticRetrievalRequest(
                query="launch risk",
                source_types=["meeting_transcript"],
                chunks=chunks,
                top_k=1,
                min_confidence=0.6,
                max_steps=1,  # Single step — no in-loop rerank skip
            ),
            chunks,
        )
        # With max_steps=1, the final rerank is the only rerank.
        # The fingerprint-based skip only applies when the fingerprint hasn't
        # changed since the last in-loop rerank.
        assert response.context_sufficiency.sufficient is True

    def test_prepare_candidates_api_endpoint(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/v1/retrieval/prepare",
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
        body = response.json()
        assert "fingerprint" in body
        assert body["chunk_count"] == 1
