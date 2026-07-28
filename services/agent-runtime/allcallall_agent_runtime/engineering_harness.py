"""One-shot engineering verification harness.

Builds a fully reproducible agent-runtime environment without standing up
MySQL / Qdrant / the Go backend:

* an in-memory SQLite checkpoint store (Module 1),
* a deterministic stub tool layer,
* an in-memory short-term memory scratchpad.

It is used both by the test-suite and by ``make agent-runtime-eval`` to
validate workflow runs and RAG retrieval quality in CI.

This module also exposes pure IR metrics (``hit_rate_at_k``,
``mean_reciprocal_rank``, ``ndcg_at_k``) that mirror the Go implementation in
``backend/cmd/agent-eval/rag_eval.go`` so Python and Go eval numbers stay
comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .checkpoint.store import SQLiteCheckpointStore
from .factory import build_agent_harness
from .providers import create_provider
from .tool_layer import StubToolLayer


# --------------------------------------------------------------------------- #
# Pure IR metrics (mirror Go cmd/agent-eval/rag_eval.go)                      #
# --------------------------------------------------------------------------- #

def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of *queries* (here, of this single query) whose top-k contains
    at least one relevant id. Binary relevance; mirrors ``ragTopKHit``."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(rid in top_k for rid in relevant) else 0.0


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant id; 0 if none in the list.
    Mirrors ``ragMRR``."""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    for idx, rid in enumerate(retrieved, start=1):
        if rid in rel_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalized DCG with binary relevance; mirrors ``ragNDCGAtK``."""
    if not relevant or not retrieved:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for idx, rid in enumerate(retrieved[:k], start=1):
        if rid in rel_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# RAG retrieval evaluation harness                                            #
# --------------------------------------------------------------------------- #

@dataclass
class RagEvalCase:
    query: str
    relevant_ids: list[str]
    # Precomputed retrieval order for deterministic fixtures. When ``None`` the
    # RagEvalHarness falls back to the injected ``retriever`` (live eval).
    retrieved: list[str] | None = None


@dataclass
class RagEvalResult:
    hit_rate_at_5: float
    mrr: float
    ndcg_at_5: float
    total: int


class RagEvalHarness:
    """Runs retrieval-quality evaluation against a labeled dataset.

    The retriever is pluggable: a deterministic mock for fixtures, or the live
    rag-runtime client for corpus evaluation (gated behind an env flag in CI).
    """

    def __init__(self, retriever: Callable[[str], list[str]] | None = None) -> None:
        self._retriever = retriever

    def _get_retrieved(self, case: RagEvalCase) -> list[str]:
        if case.retrieved is not None:
            return case.retrieved
        if self._retriever is None:
            raise ValueError("RagEvalHarness requires either case.retrieved or a retriever")
        return self._retriever(case.query)

    def evaluate(self, cases: list[RagEvalCase], k: int = 5) -> RagEvalResult:
        if not cases:
            return RagEvalResult(0.0, 0.0, 0.0, 0)
        hits = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        for case in cases:
            retrieved = self._get_retrieved(case)
            hits += hit_rate_at_k(retrieved, case.relevant_ids, k)
            mrr_sum += mean_reciprocal_rank(retrieved, case.relevant_ids)
            ndcg_sum += ndcg_at_k(retrieved, case.relevant_ids, k)
        n = len(cases)
        return RagEvalResult(hits / n, mrr_sum / n, ndcg_sum / n, n)


# --------------------------------------------------------------------------- #
# Reproducible end-to-end engineering harness                                 #
# --------------------------------------------------------------------------- #

@dataclass
class EngineeringHarness:
    """A no-external-dependency agent-runtime environment for tests / eval.

    * checkpoint store: in-memory SQLite (durable within the process,
      zero external services),
    * tool layer: deterministic stub (no Go backend),
    * short_term_memory: an in-memory scratchpad standing in for the long-lived
      GraphState between runs in a single test session.
    """

    checkpoint_store: SQLiteCheckpointStore = field(default_factory=lambda: SQLiteCheckpointStore(":memory:"))
    tool_layer: StubToolLayer = field(default_factory=StubToolLayer)
    short_term_memory: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = create_provider()
        self.harness = build_agent_harness(
            checkpoint_store=self.checkpoint_store,
            tool_layer=self.tool_layer,
            provider=self.provider,
        )

    def run_workflow(self, request: Any) -> Any:
        return self.harness.run_workflow(request)
