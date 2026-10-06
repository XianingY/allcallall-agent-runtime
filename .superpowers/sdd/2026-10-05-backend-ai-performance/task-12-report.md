# Task 12 Report: Remove Duplicate Retrieval and Slim Checkpoint State

**Implementation commit:** `08e6cec` (`perf(ai): reduce retrieval and checkpoint amplification`)
**Branch:** `perf/runtime-performance`
**Date:** 2026-10-06

## Summary

Implemented retrieval ownership modes, per-run retrieval caching, checkpoint payload projection, and RAG rerank reuse. The review-fix pass preserves checkpoint resume semantics by keeping model-typed chunk and citation lists in their serializer-native form, excludes the request-scoped RAG client, and exposes checkpoint byte sizes through a bounded Prometheus histogram.

## Changes

### Agent Runtime (`services/agent-runtime/`)

| File | Change |
|------|--------|
| `models/retrieval.py` | Added `RetrievalMode = Literal["go_context", "rag_runtime", "hybrid"]`. |
| `models/context.py` | Added retrieval mode and cache identity fields; empty/unknown retrieval mode normalizes to `hybrid` while preserving the old API input shape. |
| `state.py` | Added the per-run `retrieval_cache` state key. |
| `retrieval.py` | Added `RunRetrievalCache`, `PreparedCandidates`, `prepare_candidates()`, `rerank_prepared()`, and `compute_context_fingerprint()`. |
| `nodes/retrieval.py` | Implemented retrieval ownership, per-run cache reuse, and prepared-candidate reranking. |
| `checkpoint/payload.py` | Added `project_checkpoint_state()` and `serialized_checkpoint_size()`. Projection removes request-scoped clients/caches and compacts trace events, but preserves chunk and citation model lists needed by resumed nodes. |
| `checkpoint/mysql.py` | Integrated projection before serialization and observes original/projected sizes with `checkpoint_payload_bytes`. |
| `checkpoint/sqlite_saver.py` | Integrated the same projection before SQLite serialization. |
| `metrics.py` | Added the bounded `checkpoint_payload_bytes` histogram with stages `original` and `projected`. |

### RAG Runtime (`services/rag-runtime/`)

| File | Change |
|------|--------|
| `models.py` | Added the aligned `PreparedCandidates` and `/v1/retrieval/prepare` request model. Removed unused request-only cache fields from `RetrievalQueryRequest`. |
| `retrieval.py` | Added candidate preparation, prepared-candidate reranking, fingerprint-based duplicate rerank avoidance, and reuse of the already-ranked evidence set. |
| `api.py` | Added `POST /v1/retrieval/prepare`. |

## Tests

| Area | Coverage |
|------|----------|
| Checkpoint payload | Projection removal/retention, trace compaction, deterministic sizing, new-saver resume, pre-change fixture resume, and histogram observation. |
| Compatibility | `tests/fixtures/task12_pre_change_checkpoint.json` was generated with the serializer from base commit `bad9f58`; the new saver loads it and a graph node accesses the restored `ContextChunk`. |
| Optimization modules | Retrieval ownership, mode defaulting, cache hit/miss/eviction, candidate preparation, reranking, and fingerprints. |
| RAG runtime | Candidate preparation, prepared reranking, duplicate rerank avoidance, API routes, and eval fixtures. |

## Verification Results

All commands were run from the service directories using the repository virtual environment.

### Agent Runtime

```text
pytest tests -q
312 passed, 7 skipped, 390 warnings in 9.30s
```

The seven skips are the MySQL integration tests gated on `PY_AGENT_TEST_MYSQL_DSN`.

### RAG Runtime

```text
pytest tests/test_rag_runtime.py -q
32 passed in 0.21s
```

### Evaluation Runners

```text
python -m allcallall_agent_runtime.eval_runner --out /tmp/allcallall-agent-eval
python agent eval: 9/9 passed
```

```text
python -m allcallall_rag_runtime.eval_runner --out /tmp/allcallall-rag-eval
report summary: 3/3 cases passed
```

### Type Checking

```text
cd services/agent-runtime && python -m mypy .
Success: no issues found in 93 source files
```

```text
cd services/rag-runtime && python -m mypy .
Success: no issues found in 15 source files
```

### Lint

```text
python -m ruff check \
  services/agent-runtime/allcallall_agent_runtime \
  services/agent-runtime/tests \
  services/rag-runtime/allcallall_rag_runtime \
  services/rag-runtime/tests
All checks passed!
```

## Key Design Decisions

1. **Retrieval mode defaults to `hybrid` without breaking old requests.** The model stores the typed field as `hybrid`, and a before-validator maps empty or unknown legacy values to `hybrid`.
2. **Per-run cache is not serialized.** `retrieval_cache` is excluded from durable state; resumed executions begin with a fresh bounded cache.
3. **Checkpoint resume takes precedence over aggressive compaction.** Chunk and citation lists stay as serializer-native Pydantic objects. This avoids the reviewed `AttributeError` after resume and keeps `propose_tools`, evidence construction, and other nodes working.
4. **Old checkpoints remain readable.** The committed fixture was produced by the pre-change serializer and is loaded through the new saver in an actual interrupted-graph resume.
5. **RAG owns no dead cache API.** The agent owns context fingerprint/corpus-version cache identity, so those fields are not exposed by RAG requests.
6. **Duplicate reranking is avoided in RAG.** Unchanged fingerprints reuse the previous ranked list, and evidence-pack construction accepts the already-ranked chunks instead of reranking again.
