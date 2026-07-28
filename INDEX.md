# Documentation Index — allcallall-agent-runtime

Repo-local index for the standalone Python Agent + RAG runtime. The
cross-repository master index (covering the Go/Web/Mobile main repo as well)
lives in the main repo: `AllCallAll/INDEX.md`.

## Getting Started

| Title | Path |
| --- | --- |
| Repository overview, quick start, APIs | [`README.md`](README.md) |
| Agent Runtime service guide | [`services/agent-runtime/README.md`](services/agent-runtime/README.md) |
| RAG Runtime service guide | [`services/rag-runtime/README.md`](services/rag-runtime/README.md) |

## Architecture & Design

| Title | Path |
| --- | --- |
| System architecture, workflows, safety model | [`docs/architecture.md`](docs/architecture.md) |
| Three-layer harness decoupling (scheduling / persistence / tool) | [`docs/harness-architecture.md`](docs/harness-architecture.md) |
| Loop engineering: bounded role loops and loop contract | [`docs/loop-engineering.md`](docs/loop-engineering.md) |
| Two-tier CheckAgent quality/safety loop | [`docs/check-agents.md`](docs/check-agents.md) |
| Hierarchical memory and context compression | [`docs/context-compression.md`](docs/context-compression.md) |

## Skills & Tools

| Title | Path |
| --- | --- |
| Skill catalog + dynamic registry with SecurityOverlay | [`docs/skill-registry.md`](docs/skill-registry.md) |
| MCP tool descriptors and async tool queue | [`docs/mcp-tools-async-queue.md`](docs/mcp-tools-async-queue.md) |
| Go Tool Bridge HTTP protocol | [`docs/tool-bridge-protocol.md`](docs/tool-bridge-protocol.md) |

## Configuration & Integration

| Title | Path |
| --- | --- |
| Full environment variable reference | [`docs/configuration.md`](docs/configuration.md) |
| Cross-service wiring with the AllCallAll Go backend | [`docs/allcallall-integration.md`](docs/allcallall-integration.md) |
| Contracts: generated schemas and golden fixtures | [`contracts/README.md`](contracts/README.md) |

## Evaluation & Metrics

| Title | Path |
| --- | --- |
| Eval methodology and current evidence | [`docs/eval-methodology.md`](docs/eval-methodology.md) |
| Engineering harness and IR metrics (HitRate@5 / MRR / NDCG) | [`docs/engineering-harness.md`](docs/engineering-harness.md) |
| Resume-safe agent metrics wording | [`docs/resume-agent-metrics.md`](docs/resume-agent-metrics.md) |
| Manual pilot UX sample (illustrative only) | [`docs/manual-pilot-ux-sample.md`](docs/manual-pilot-ux-sample.md) |
| Generated portfolio eval report (machine-generated) | [`docs/generated-ai-agent-portfolio-eval/portfolio-eval.md`](docs/generated-ai-agent-portfolio-eval/portfolio-eval.md) |

## Conventions

- English H1 titles, sentence-case section headings, fenced code blocks with
  language tags, `-` bullets, and tables for reference material.
- `docs/generated-*` directories are machine-generated; regenerate via
  `make portfolio-eval` instead of editing by hand.
- Configuration values are documented once in `docs/configuration.md`; other
  documents link there rather than duplicating tables.
