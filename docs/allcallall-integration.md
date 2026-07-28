# AllCallAll Integration

AllCallAll integrates this runtime over HTTP.

Backend environment:

```bash
AGENT_RUNTIME=python_langgraph
PY_AGENT_RUNTIME_BASE_URL=http://agent-runtime:8090
PY_RAG_RUNTIME_BASE_URL=http://rag-runtime:8091
PY_AGENT_RUNTIME_STRICT=true
AGENT_RUNTIME_TOOL_TOKEN=<shared-token>
```

Agent Runtime environment:

```bash
PY_AGENT_PROVIDER=rules
PY_AGENT_TOOL_BRIDGE_BASE_URL=http://backend:8080
PY_AGENT_TOOL_BRIDGE_TOKEN=<shared-token>
PY_RAG_RUNTIME_BASE_URL=http://rag-runtime:8091
```

RAG Runtime environment:

```bash
PY_RAG_RERANK_PROVIDER=rules
PY_RAG_TOOL_BRIDGE_BASE_URL=http://backend:8080
PY_RAG_TOOL_BRIDGE_TOKEN=<shared-token>
```

For local development inside the AllCallAll monorepo, use `examples/allcallall-compose.override.yml` from this repository as the compose override. The main repo expects this repository as a sibling directory: `../allcallall-agent-runtime`.

The snippets above are the minimal cross-service wiring. The full variable
reference (including checkpointing, quality-loop, and context-compression
settings) lives in `docs/configuration.md`; the Go backend's own variables are
documented in the main repo's `docs/configuration/configuration.md`.

