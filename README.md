# AllCallAll Agent Runtime

Standalone Python Agent and RAG runtime for AllCallAll.

The runtime is intentionally separated from the AllCallAll Go backend:

- Python owns Agent orchestration, LangGraph workflows, bounded ReAct loops, prompt/provider adapters, Agentic RAG, rerank, grounding checks, traces, citations, tool proposals, and deterministic eval.
- Go remains the product source of truth for users, organizations, conversations, meetings, transcripts, permissions, approvals, audit logs, and write execution.

## Repository Layout

- `services/agent-runtime`: FastAPI + LangGraph Agent Runtime.
- `services/rag-runtime`: FastAPI Agentic RAG / rerank / grounding service.
- `packages/shared`: shared Pydantic models and scoring utilities.
- `packages/sdk`: typed Python client SDK for both services.
- `contracts`: generated JSON Schemas and golden JSON fixtures.
- `examples`: Docker Compose and curl examples.
- `docs`: architecture, protocol, skill, eval, and AllCallAll integration notes.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
make install-dev
make verify
```

Run services locally:

```bash
make run-agent-runtime
make run-rag-runtime
```

Or use Docker:

```bash
docker compose -f examples/docker-compose.yml up --build
```

Tagged releases publish images to GitHub Container Registry:

```text
ghcr.io/xianingy/allcallall-agent-runtime/agent-runtime:v0.1.0
ghcr.io/xianingy/allcallall-agent-runtime/rag-runtime:v0.1.0
```

## Runtime APIs

Agent Runtime:

- `GET /health`
- `GET /ready`
- `GET /v1/capabilities`
- `POST /v1/agents/react/run`
- `POST /v1/workflows/{preset}/run`

RAG Runtime:

- `GET /health`
- `GET /ready`
- `POST /v1/retrieval/query`
- `POST /v1/retrieval/rerank`
- `POST /v1/retrieval/agentic`
- `POST /v1/grounding/check`

## Safety Boundary

The runtime never writes AllCallAll business data directly. Read skills may call the Go Tool Bridge. Write skills are returned as approval-required proposals; the Go backend validates, audits, and executes them only after approval.

## Eval

```bash
make agent-eval
make rag-eval
```

The eval suite is deterministic regression evidence for task completion, citation grounding, approval safety, retrieval refinement, rerank, and insufficient-context handling. It is not an open-domain model-quality benchmark.
