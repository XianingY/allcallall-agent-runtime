# Configuration Reference

Single source of truth for all runtime environment variables. Service READMEs and
`docs/allcallall-integration.md` link here instead of duplicating tables.

All values are loaded through `pydantic-settings`:

- Agent Runtime: prefix `PY_AGENT_` (`services/agent-runtime/allcallall_agent_runtime/config.py`)
- RAG Runtime: prefix `PY_RAG_` (`services/rag-runtime/allcallall_rag_runtime/config.py`)

## Agent Runtime (`PY_AGENT_*`)

### Provider

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_PROVIDER` | `rules` | Planner provider: `rules` (deterministic) or `openai_compatible`. |
| `PY_AGENT_PROVIDER_STRICT` | `true` | Fail fast on provider errors instead of silent fallback. |
| `PY_AGENT_OPENAI_BASE_URL` | empty | OpenAI-compatible endpoint base URL. |
| `PY_AGENT_OPENAI_API_KEY` | empty | API key for the endpoint. |
| `PY_AGENT_OPENAI_MODEL` | `gpt-4` | Model name. |
| `PY_AGENT_OPENAI_TIMEOUT_SEC` | `30.0` | Provider request timeout. |

### Tool Bridge and RAG Runtime

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_TOOL_BRIDGE_BASE_URL` | empty | Go backend Tool Bridge base URL. |
| `PY_AGENT_TOOL_BRIDGE_TOKEN` | empty | Shared bearer token (`AGENT_RUNTIME_TOOL_BRIDGE_TOKEN` on the Go side). |
| `PY_AGENT_TOOL_BRIDGE_TIMEOUT_SEC` | `10.0` | Tool Bridge request timeout. |
| `PY_AGENT_RAG_RUNTIME_BASE_URL` | empty | RAG Runtime base URL. |
| `PY_AGENT_RAG_RUNTIME_TIMEOUT_SEC` | `10.0` | RAG Runtime request timeout. |

### Agentic RAG

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_ENABLE_AGENTIC_RAG` | `false` | Enable bounded multi-step retrieval refinement. |
| `PY_AGENT_RAG_MAX_RETRIEVAL_STEPS` | `3` | Retrieval refinement budget. |
| `PY_AGENT_RAG_MIN_CONFIDENCE` | `0.6` | Minimum confidence before stopping refinement. |

### Retries

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_PROVIDER_MAX_RETRIES` | `2` | Provider call retries. |
| `PY_AGENT_TOOL_BRIDGE_MAX_RETRIES` | `2` | Tool Bridge call retries. |
| `PY_AGENT_RAG_RUNTIME_MAX_RETRIES` | `2` | RAG Runtime call retries. |
| `PY_AGENT_RETRY_BASE_DELAY_SEC` | `0.5` | Exponential backoff base delay. |
| `PY_AGENT_RETRY_MAX_DELAY_SEC` | `8.0` | Backoff delay cap. |

### Checkpointing (persistence layer)

See `docs/harness-architecture.md` for the pluggable `CheckpointStore` design.

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_CHECKPOINT_STORE` | empty | `""` (auto) \| `none` \| `mysql` \| `sqlite` \| `memory`. Auto resolves to MySQL when enabled, else no checkpointing. |
| `PY_AGENT_CHECKPOINT_MYSQL_ENABLED` | `false` | Legacy switch for the MySQL checkpoint saver (used by auto mode). |
| `PY_AGENT_CHECKPOINT_MYSQL_DSN` | empty | MySQL DSN for `MySQLCheckpointSaver`. |
| `PY_AGENT_CHECKPOINT_SQLITE_PATH` | `:memory:` | SQLite file path, or `:memory:` for reproducible test environments. |

### Workflow harness and quality loop

See `docs/check-agents.md` for the two-tier CheckAgent loop.

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_HARNESS` | `allcallall_v1` | Harness implementation selector. |
| `PY_AGENT_LOOP_MAX_STEPS` | `5` | Bounded loop step budget. |
| `PY_AGENT_ENABLE_CRITIC` | `true` | Enable the critic node. |
| `PY_AGENT_ENABLE_MEMORY_REFLECTION` | `true` | Enable MemoryAgent reflection proposals. |
| `PY_AGENT_MAX_QUALITY_RETRIES` | `1` | L1 `quality_check` revise budget before escalation. |
| `PY_AGENT_PROMPT_VERSION` | empty | Prompt version pin. |
| `PY_AGENT_ENABLE_GROUNDING_CHECK` | `false` | Enable citation-grounding verification. |

### Context compression (hierarchical memory)

See `docs/context-compression.md`.

| Variable | Default | Description |
| --- | --- | --- |
| `PY_AGENT_ENABLE_CONTEXT_COMPRESSION` | `false` | Opt-in; off keeps legacy request behavior. |
| `PY_AGENT_CONTEXT_COMPRESSION_STRATEGY` | `summary` | `summary` (extractive) \| `full` (recent-only). |
| `PY_AGENT_MODEL_HISTORY_MAX_TOKENS` | `4000` | Token budget for compressed model history. |

## RAG Runtime (`PY_RAG_*`)

### Tool Bridge

| Variable | Default | Description |
| --- | --- | --- |
| `PY_RAG_TOOL_BRIDGE_BASE_URL` | empty | Go backend Tool Bridge base URL. |
| `PY_RAG_TOOL_BRIDGE_TOKEN` | empty | Shared bearer token. |
| `PY_RAG_TOOL_BRIDGE_TIMEOUT_SEC` | `10.0` | Request timeout. |

### Retrieval and rerank

| Variable | Default | Description |
| --- | --- | --- |
| `PY_RAG_RERANK_PROVIDER` | `rules` | Rerank provider. |
| `PY_RAG_TOP_K` | `8` | Retrieval result count. |
| `PY_RAG_MAX_STEPS` | `3` | Agentic retrieval step budget. |
| `PY_RAG_MIN_CONFIDENCE` | `0.6` | Stop-condition confidence threshold. |
| `PY_RAG_ENABLE_GRAPH_EXPANSION` | `true` | Knowledge-graph query expansion. |
| `PY_RAG_ENABLE_LLAMAINDEX_BASELINE` | `false` | Optional LlamaIndex comparison baseline. |

### Optional vector store

Production AllCallAll still authorizes retrieval via the Go backend; the Qdrant
adapter is optional.

| Variable | Default | Description |
| --- | --- | --- |
| `PY_RAG_VECTOR_STORE` | `none` | `none` \| `qdrant`. |
| `PY_RAG_QDRANT_URL` | empty | Qdrant endpoint. |
| `PY_RAG_QDRANT_COLLECTION` | `allcallall_context_chunks` | Collection name. |
| `PY_RAG_QDRANT_API_KEY` | empty | API key. |
| `PY_RAG_QDRANT_TIMEOUT_SEC` | `5.0` | Request timeout. |

## Go Backend Pairing

The Go backend (main AllCallAll repo) reads its own `AGENT_*` / `PY_*` pairing
variables; see `docs/allcallall-integration.md` for the cross-service wiring and
the main repo's `docs/configuration/configuration.md` for the full backend list.
