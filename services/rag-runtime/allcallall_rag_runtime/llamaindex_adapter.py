"""Eval-only LlamaIndex retrieval adapter.

The production AllCallAll path still retrieves through Go authorization and the configured
RAG runtime. This module gives interview/eval fixtures a baseline shaped like a LlamaIndex
retriever without requiring LlamaIndex to be installed in the default runtime image.
"""

from __future__ import annotations

import importlib.util

from pydantic import BaseModel, Field
from shared.scoring import tokenize

from .models import ContextChunk
from .retrieval import rerank


class LlamaIndexBaselineResult(BaseModel):
    provider: str = "llamaindex_eval_baseline"
    available: bool = False
    hits: list[ContextChunk] = Field(default_factory=list)


def run_fixture_retrieval(query: str, chunks: list[ContextChunk], top_k: int = 5) -> LlamaIndexBaselineResult:
    """Run an eval-only baseline retrieval path.

    If LlamaIndex is installed, callers can replace this implementation with a real
    VectorStoreIndex flow. The default deterministic path keeps CI and resume metrics
    reproducible without external embedding services.
    """
    available = llamaindex_available()
    if available:
        # Keep deterministic ranking for now; the optional dependency is only a framework
        # compatibility signal until an embedding provider is explicitly configured.
        return LlamaIndexBaselineResult(available=True, hits=rerank(query, chunks, top_k).chunks)
    tokens = set(tokenize(query, remove_stopwords=True))
    scored = [
        (
            len(tokens.intersection(tokenize(chunk.snippet, remove_stopwords=True))),
            chunk,
        )
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return LlamaIndexBaselineResult(hits=[chunk for _, chunk in scored[: max(1, top_k)]])


def llamaindex_available() -> bool:
    try:
        return importlib.util.find_spec("llama_index.core") is not None
    except ModuleNotFoundError:
        return False
