# AllCallAll RAG Runtime

Python FastAPI runtime for Agent retrieval orchestration, rerank, grounding checks, and deterministic RAG eval.

The service does not directly access AllCallAll business databases. In production it calls the Go backend internal retrieval bridge with `PY_RAG_TOOL_BRIDGE_BASE_URL` and `PY_RAG_TOOL_BRIDGE_TOKEN`; in eval/tests it can operate on inline fixture chunks.

It supports more than ordinary top-k RAG:

- Dynamic query routing into `chat`, `consult`, and `risk`.
- Adaptive retrieval strategies: `single_pass`, `graph_augmented`, and `multi_hop`.
- Lightweight knowledge-graph edge extraction from retrieved evidence.
- Evidence packs with citations, source coverage, context sufficiency, and grounding checks.

## Run Locally

```bash
cd rag-runtime
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
uvicorn allcallall_rag_runtime.main:app --reload --port 8091
```

## Configuration

All settings use the `PY_RAG_` prefix (`allcallall_rag_runtime/config.py`).
Key variables: `PY_RAG_TOOL_BRIDGE_BASE_URL` / `PY_RAG_TOOL_BRIDGE_TOKEN`
(Go retrieval bridge), `PY_RAG_TOP_K`, `PY_RAG_MAX_STEPS`,
`PY_RAG_MIN_CONFIDENCE`, `PY_RAG_ENABLE_GRAPH_EXPANSION`, and the optional
Qdrant adapter (`PY_RAG_VECTOR_STORE=qdrant`, `PY_RAG_QDRANT_*`).

Full reference: `../../docs/configuration.md`.

## API

- `GET /health`
- `GET /ready`
- `GET /v1/capabilities`
- `POST /v1/retrieval/query`
- `POST /v1/retrieval/rerank`
- `POST /v1/retrieval/agentic`
- `POST /v1/grounding/check`

## Eval

```bash
python -m allcallall_rag_runtime.eval_runner --out evals/reports
```

The eval is deterministic fixture evidence for retrieval refinement, rerank ordering, grounding, and insufficient-context behavior. It is not an open-domain RAG quality claim.

The fixture report also tracks route match and graph expansion success for the dynamic RAG path.
