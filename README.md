# AllCallAll Agent Runtime

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

AllCallAll Agent Runtime is the standalone Python orchestration and retrieval
layer for the [AllCallAll](https://github.com/XianingY/allcallall) collaboration
platform.

## Why a Separate Runtime

Model orchestration, retrieval experiments, and evaluation evolve at a
different pace from product data and authorization. Keeping them in a separate
repository lets the Python services be built, tested, and released
independently while preserving a narrow contract with the Go platform.

This repository does not implement the AllCallAll product UI or own durable
business writes. Those responsibilities remain in the main product repository.

## Services and Packages

| Path | Responsibility |
| --- | --- |
| `services/agent-runtime/` | FastAPI and LangGraph Agent orchestration service |
| `services/rag-runtime/` | Retrieval planning, reranking, evidence packs, and grounding service |
| `services/sandbox-runner/` | Isolated execution worker and supervisor transport |
| `services/reference-mcp/` | Deterministic HTTPS MCP reference service with legacy import compatibility |
| `packages/shared/` | Shared Pydantic contracts, scoring, and runtime utilities |
| `packages/sdk/` | Typed Python client for Agent and RAG services |
| `contracts/` | Generated JSON Schemas and golden fixtures |
| `examples/` | Local Docker Compose and request examples |

## Safety Boundary

The runtime may read authorized context through the Go Tool Bridge. It never
writes AllCallAll business data directly. Write-capable tools produce
approval-required proposals; the Go backend validates permissions, records the
audit trail, and executes approved writes.

Go remains authoritative for users, organizations, conversations, meetings,
transcripts, permissions, approvals, and audit logs. Python owns orchestration,
retrieval, grounding, citations, traces, and evaluation.

## Quick Start

Python 3.11 or newer is required; CI uses Python 3.12.

```bash
python3 -m venv .venv
. .venv/bin/activate
make install-dev
make test
make lint
make typecheck
make contracts-check
```

Run the local services:

```bash
make run-agent-runtime
make run-rag-runtime
```

Or use the example Compose topology:

```bash
docker compose -f examples/docker-compose.yml up --build
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

See the service READMEs and generated contracts for request and response
details.

## Evaluation

```bash
make agent-eval
make rag-eval
```

Evaluation results are deterministic regression evidence for the checked
fixtures, including task completion, grounding, approval safety, retrieval
refinement, reranking, and insufficient-context handling. They are not claims
about open-domain model quality. See the [evaluation methodology](docs/evaluation/methodology.md)
and [engineering harness](docs/evaluation/engineering-harness.md) for fixture scope,
commands, and interpretation.

## Documentation

- [Documentation index](docs/README.md)
- [Quick start](docs/getting-started/quick-start.md)
- [Architecture](docs/architecture/overview.md)
- [Harness architecture](docs/architecture/harness.md)
- [Configuration](docs/reference/configuration.md)
- [Tool Bridge protocol](docs/reference/tool-bridge-protocol.md)
- [AllCallAll integration](docs/guides/allcallall-integration.md)
- [Contract governance](contracts/README.md)

`INDEX.md` remains as a compatibility pointer for older links.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Keep public imports, HTTP routes,
environment variables, JSON Schemas, and generated contracts compatible unless
an approved change includes an explicit migration path.

## Security and Support

Do not report vulnerabilities in public issues. Follow [SECURITY.md](SECURITY.md)
for private disclosure. For usage questions, reproducible bugs, and design
proposals, see [SUPPORT.md](SUPPORT.md).

## License

AllCallAll Agent Runtime is available under the [MIT License](LICENSE).
