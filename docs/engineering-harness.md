# Engineering Harness and RAG Eval Metrics

`engineering_harness.py` provides a fully deterministic, dependency-free test
bed for the runtime, plus standard IR metrics mirrored from the Go evaluator
(`backend/cmd/agent-eval/rag_eval.go` in the main repo) so both stacks report
comparable numbers.

## Components

- `EngineeringHarness` — assembles a real `AllCallAllAgentHarness` from
  in-memory seams: SQLite (`:memory:`) checkpoint store, `StubToolLayer`, and
  in-memory short-term memory. End-to-end workflow runs are reproducible with
  no external services.
- `RagEvalHarness` — runs retrieval eval cases against a pluggable retriever
  callable and scores them with the IR metrics below.
- `RagEvalCase` / `RagEvalResult` — dataclasses describing fixture cases and
  aggregate results.

## IR Metrics

Pure functions, exact-match semantics:

| Metric | Function | Notes |
| --- | --- | --- |
| HitRate@K | `hit_rate_at_k` | 1 if any relevant id appears in the top-K results. |
| MRR | `mean_reciprocal_rank` | Reciprocal of the first relevant rank. Not capped at K: a hit at rank 6 still contributes 1/6 even when K=5. |
| NDCG@K | `ndcg_at_k` | Binary-relevance DCG normalized by ideal DCG. |

## Current Reference Numbers

The 120-case fixture in `tests/test_engineering_harness.py` locks in:

- `HitRate@5 = 0.9667` (116/120 cases hit in top-5)
- `MRR = 0.9083` (102 rank-1 hits + 14 rank-2 hits: `(102*1 + 14*0.5)/120`)

These are deterministic regression anchors, not open-domain quality claims —
the same methodology caveats as `docs/eval-methodology.md` apply.

## Usage

```python
from allcallall_agent_runtime.engineering_harness import (
    EngineeringHarness, RagEvalHarness, RagEvalCase,
)

harness = EngineeringHarness().build()      # ready-to-run agent harness
result = RagEvalHarness(retriever).run(cases, k=5)
print(result.hit_rate_at_k, result.mrr)
```

## Tests

`tests/test_engineering_harness.py` (5 cases): metric exactness against
hand-computed values, the 120-case fixture producing the reference numbers,
and a reproducible end-to-end workflow run through the in-memory harness.
