from .api import (
    capabilities,
    create_app,
    grounding,
    health,
    prometheus_metrics,
    ready,
    retrieval_agentic,
    retrieval_query,
    retrieval_rerank,
)

app = create_app()

__all__ = [
    "app",
    "capabilities",
    "create_app",
    "grounding",
    "health",
    "prometheus_metrics",
    "ready",
    "retrieval_agentic",
    "retrieval_query",
    "retrieval_rerank",
]
