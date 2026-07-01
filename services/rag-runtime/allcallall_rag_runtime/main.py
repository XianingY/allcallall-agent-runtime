from __future__ import annotations

from fastapi import FastAPI, Response

from .go_bridge import GoRetrievalBridge
from .metrics import metrics
from .models import (
    AgenticRetrievalRequest,
    AgenticRetrievalResponse,
    ContextChunk,
    GroundingCheckRequest,
    GroundingCheckResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RerankRequest,
    RerankResponse,
)
from .qdrant_adapter import QdrantAdapter, QdrantAdapterError
from .retrieval import agentic_retrieve, filter_chunks, grounding_check, rerank

app = FastAPI(title="AllCallAll RAG Runtime", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "python_rag"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/v1/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "runtime": "python_rag",
        "retrieval": ["query", "rerank", "agentic", "grounding_check"],
        "intent_routes": ["chat", "consult", "risk"],
        "strategies": [
            "single_pass",
            "adaptive",
            "graph_augmented",
            "multi_hop",
            "risk_focused",
            "no_retrieval",
        ],
        "vector_stores": ["inline", "go_bridge", "qdrant_optional"],
        "evidence": ["citations", "context_sufficiency", "knowledge_graph_edges"],
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(metrics.prometheus(), media_type="text/plain; version=0.0.4")


@app.post("/v1/retrieval/query", response_model=RetrievalQueryResponse)
def retrieval_query(request: RetrievalQueryRequest) -> RetrievalQueryResponse:
    metrics.inc("rag_runtime_query_total")
    bridge = GoRetrievalBridge()
    bridge_chunks = bridge.query(request) if bridge.configured() else []
    qdrant_chunks: list[ContextChunk] = []
    if not bridge_chunks:
        try:
            qdrant_chunks = QdrantAdapter().query(request)
        except QdrantAdapterError:
            qdrant_chunks = []
    source = "go_bridge" if bridge_chunks else "qdrant" if qdrant_chunks else "inline"
    chunks = bridge_chunks or qdrant_chunks or request.chunks
    scoped = filter_chunks(chunks, request.source_types)[: max(1, request.top_k)]
    return RetrievalQueryResponse(query=request.query, chunks=scoped, count=len(scoped), source=source)


@app.post("/v1/retrieval/rerank", response_model=RerankResponse)
def retrieval_rerank(request: RerankRequest) -> RerankResponse:
    metrics.inc("rag_runtime_rerank_total")
    return rerank(request.query, request.chunks, request.top_k)


@app.post("/v1/retrieval/agentic", response_model=AgenticRetrievalResponse)
def retrieval_agentic(request: AgenticRetrievalRequest) -> AgenticRetrievalResponse:
    metrics.inc("rag_runtime_agentic_total")
    bridge = GoRetrievalBridge()
    bridge_chunks = bridge.query(request) if bridge.configured() else []
    qdrant_chunks: list[ContextChunk] = []
    if not bridge_chunks:
        try:
            qdrant_chunks = QdrantAdapter().query(request)
        except QdrantAdapterError:
            qdrant_chunks = []
    chunks = bridge_chunks or qdrant_chunks or request.chunks
    source = "go_bridge" if bridge_chunks else "qdrant" if qdrant_chunks else "inline"
    response = agentic_retrieve(request, chunks)
    return response.model_copy(update={"vector_store": source})


@app.post("/v1/grounding/check", response_model=GroundingCheckResponse)
def grounding(request: GroundingCheckRequest) -> GroundingCheckResponse:
    metrics.inc("rag_runtime_grounding_check_total")
    return grounding_check(request.answer, request.citations)
