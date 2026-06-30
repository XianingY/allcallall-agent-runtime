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

For local development inside the AllCallAll monorepo, use `examples/allcallall-compose.override.yml` from this repository as the compose override.

