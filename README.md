# AllCallAll Agent Runtime

Standalone Python Agent and RAG runtime for AllCallAll.

The runtime is intentionally separated from the AllCallAll Go backend:

- Python owns Agent orchestration, LangGraph workflows, bounded ReAct loops, prompt/provider adapters, Agentic RAG, rerank, grounding checks, traces, citations, tool proposals, and deterministic eval.
- Go remains the product source of truth for users, organizations, conversations, meetings, transcripts, permissions, approvals, audit logs, and write execution.

The runtime is designed as a production-grade Agent Runtime Harness rather than a simple MCP/RAG/function-calling demo. It now includes dynamic CHAT/CONSULT/RISK routing, knowledge-graph query expansion, adaptive multi-hop RAG, MemoryAgent reflection, RiskGuardian-style assessment, approval-gated async tool queue metadata, and deterministic eval evidence.

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
- `GET /v1/capabilities`
- `POST /v1/retrieval/query`
- `POST /v1/retrieval/rerank`
- `POST /v1/retrieval/agentic`
- `POST /v1/grounding/check`

## Runtime Harness Capabilities

- Dynamic intent routing chooses `chat`, `consult`, or `risk` before retrieval and records the route in responses and traces.
- Agentic RAG uses bounded retrieval refinement, source-scope planning, rerank, evidence packs, context sufficiency, and citation grounding.
- Knowledge-graph expansion infers lightweight evidence edges from retrieved chunks and injects expanded terms into retrieval attempts.
- Multi-agent workflow roles include Searcher, MemoryAgent, Summarizer, and RiskGuardian-style risk assessment under a supervisor trace.
- Write tools remain proposal-only, but proposals now carry async queue, retry, rate-limit, idempotency, and dead-letter metadata for Go-side execution.

## Safety Boundary

The runtime never writes AllCallAll business data directly. Read skills may call the Go Tool Bridge. Write skills are returned as approval-required proposals; the Go backend validates, audits, and executes them only after approval.

## Eval

```bash
make agent-eval
make rag-eval
```

The eval suite is deterministic regression evidence for task completion, citation grounding, approval safety, retrieval refinement, rerank, and insufficient-context handling. It is not an open-domain model-quality benchmark. IR-metric anchors: `HitRate@5 = 0.9667`, `MRR = 0.9083` (see `docs/engineering-harness.md`).

## Documentation

Full index: [`INDEX.md`](INDEX.md). Key documents:

| Document | Content |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | System architecture, workflows, safety model. |
| [`docs/harness-architecture.md`](docs/harness-architecture.md) | Three-layer harness decoupling (scheduling / persistence / tool). |
| [`docs/loop-engineering.md`](docs/loop-engineering.md) | Bounded role loops and the loop contract. |
| [`docs/check-agents.md`](docs/check-agents.md) | Two-tier CheckAgent quality/safety loop. |
| [`docs/context-compression.md`](docs/context-compression.md) | Hierarchical memory and token-bounded model history. |
| [`docs/skill-registry.md`](docs/skill-registry.md) | Skill catalog + dynamic registry with SecurityOverlay. |
| [`docs/mcp-tools-async-queue.md`](docs/mcp-tools-async-queue.md) | MCP tool descriptors and async tool queue semantics. |
| [`docs/engineering-harness.md`](docs/engineering-harness.md) | Deterministic engineering harness and IR metrics. |
| [`docs/eval-methodology.md`](docs/eval-methodology.md) | Eval scope and current evidence. |
| [`docs/configuration.md`](docs/configuration.md) | Full environment variable reference. |
| [`docs/tool-bridge-protocol.md`](docs/tool-bridge-protocol.md) | Go Tool Bridge HTTP protocol. |
| [`docs/allcallall-integration.md`](docs/allcallall-integration.md) | Cross-service wiring with the Go backend. |
