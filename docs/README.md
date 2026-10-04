# AllCallAll Agent Runtime Documentation

This is the canonical documentation index for the standalone Python Agent and
RAG runtime. Product behavior, authorization, and durable writes are documented
in the [AllCallAll repository](https://github.com/XianingY/allcallall).

English documentation is authoritative. Historical plans, reports, and
portfolio material are retained under the archive and are not current runtime
guidance.

## Getting Started

- [Quick Start](getting-started/quick-start.md) — install, verify, and run the local services.
- [Repository Overview](../README.md) — scope, safety boundary, APIs, and contribution links.

## Architecture

- [Architecture Overview](architecture/overview.md) — service boundary and primary workflows.
- [Harness Architecture](architecture/harness.md) — scheduling, persistence, and tool-layer seams.
- [Loop Engineering](architecture/loop-engineering.md) — bounded role loops and stop conditions.
- [Two-Tier CheckAgent Loop](architecture/check-agents.md) — quality and safety decisions.
- [Context Compression](architecture/context-compression.md) — token-bounded history and memory layers.
- [Skill Registry](architecture/skill-registry.md) — capability manifests and security overlays.
- [MCP Tools and Async Queue](architecture/mcp-tools-async-queue.md) — tool wrapping and queued-write semantics.

## Guides

- [AllCallAll Integration](guides/allcallall-integration.md) — connect the Python services to the Go product backend.

## Reference

- [Configuration](reference/configuration.md) — canonical `PY_AGENT_*` and `PY_RAG_*` settings.
- [Tool Bridge Protocol](reference/tool-bridge-protocol.md) — authenticated read and retrieval bridge APIs.
- [Contract Governance](../contracts/README.md) — generated schemas and compatibility checks.

## Evaluation

- [Evaluation Methodology](evaluation/methodology.md) — fixture scope, metrics, and interpretation limits.
- [Engineering Harness](evaluation/engineering-harness.md) — deterministic end-to-end and IR test harnesses.
- [Badcase, SFT, and Online Evaluation](evaluation/badcase-sft-online-eval.md) — feedback and candidate-model evaluation loop.

## Service Documentation

- [Agent Runtime](../services/agent-runtime/README.md)
- [RAG Runtime](../services/rag-runtime/README.md)

## Archive

[Archived documentation](archive/README.md) contains historical portfolio
material, generated or point-in-time reports, and superseded plans.
