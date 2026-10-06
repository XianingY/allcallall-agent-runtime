from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request, Response

from .clients import RAGClients, build_rag_clients
from .config import config as rag_config
from .metrics import metrics
from .models import (
    AgenticRetrievalRequest,
    AgenticRetrievalResponse,
    GroundingCheckRequest,
    GroundingCheckResponse,
    PreparedCandidates,
    PrepareCandidatesRequest,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RerankRequest,
    RerankResponse,
)
from .pipeline import select_retrieval_chunks
from .retrieval import agentic_retrieve, filter_chunks, grounding_check, prepare_candidates, rerank


router = APIRouter()


def _get_clients(request: Request) -> RAGClients:
    """Return the process-owned client bundle from application state."""
    clients: RAGClients = request.app.state.clients
    return clients


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "python_rag"}


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/v1/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "runtime": "python_rag",
        "retrieval": ["query", "rerank", "prepare", "agentic", "grounding_check"],
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


@router.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(metrics.prometheus(), media_type="text/plain; version=0.0.4")


@router.post("/v1/retrieval/query", response_model=RetrievalQueryResponse)
def retrieval_query(
    request: RetrievalQueryRequest,
    clients: RAGClients = Depends(_get_clients),
) -> RetrievalQueryResponse:
    metrics.inc("rag_runtime_query_total")
    chunks, source = select_retrieval_chunks(request, clients=clients)
    scoped = filter_chunks(chunks, request.source_types)[: max(1, request.top_k)]
    return RetrievalQueryResponse(query=request.query, chunks=scoped, count=len(scoped), source=source)


@router.post("/v1/retrieval/rerank", response_model=RerankResponse)
def retrieval_rerank(request: RerankRequest) -> RerankResponse:
    metrics.inc("rag_runtime_rerank_total")
    return rerank(request.query, request.chunks, request.top_k)


@router.post("/v1/retrieval/prepare", response_model=PreparedCandidates)
def retrieval_prepare(request: PrepareCandidatesRequest) -> PreparedCandidates:
    """Prepare candidates for later reranking.

    Returns a fingerprint that callers can use to skip redundant reranking
    when the candidate set has not changed.
    """
    metrics.inc("rag_runtime_prepare_total")
    return prepare_candidates(request.query, request.chunks, request.source_types)


@router.post("/v1/retrieval/agentic", response_model=AgenticRetrievalResponse)
def retrieval_agentic(
    request: AgenticRetrievalRequest,
    clients: RAGClients = Depends(_get_clients),
) -> AgenticRetrievalResponse:
    metrics.inc("rag_runtime_agentic_total")
    chunks, source = select_retrieval_chunks(request, clients=clients)
    response = agentic_retrieve(request, chunks)
    return response.model_copy(update={"vector_store": source})


@router.post("/v1/grounding/check", response_model=GroundingCheckResponse)
def grounding(request: GroundingCheckRequest) -> GroundingCheckResponse:
    metrics.inc("rag_runtime_grounding_check_total")
    return grounding_check(request.answer, request.citations)


def create_app() -> FastAPI:
    """Create the RAG FastAPI application with its public routes."""
    application = FastAPI(title="AllCallAll RAG Runtime", version="0.1.0", lifespan=_lifespan)
    application.include_router(router)
    return application


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Own the process-lifetime RAG client bundle."""
    clients = build_rag_clients(rag_config)
    application.state.clients = clients
    try:
        yield
    finally:
        clients.close()
