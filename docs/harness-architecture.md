# Harness Architecture: Three-Layer Decoupling

The Agent Runtime Harness separates three concerns so each can evolve and be
tested independently:

1. Scheduling layer — `AllCallAllAgentHarness` (`harness.py`): request
   normalization, LangGraph dispatch, trace/citation projection.
2. Persistence layer — `CheckpointStore` protocol (`checkpoint/store.py`):
   pluggable LangGraph checkpointing.
3. Tool layer — `ToolLayer` protocol (`tool_layer.py`): produces the tool
   bridge a workflow run should use.

```text
AllCallAllAgentHarness
  |-- checkpoint_store: CheckpointStore   (persistence, injectable)
  |-- tool_layer:       ToolLayer         (tool bridge production, injectable)
  `-- provider:         Planner provider  (rules / openai_compatible)
```

## Persistence Layer: `CheckpointStore`

`checkpoint/store.py` defines the protocol plus four implementations:

| Implementation | Backing | Use case |
| --- | --- | --- |
| `NullCheckpointStore` | none | Default when checkpointing is disabled. |
| `MemoryCheckpointStore` | in-process | Unit tests, ephemeral runs. |
| `SQLiteCheckpointStore` | SQLite file or `:memory:` | Local development, reproducible tests. |
| `MySQLCheckpointStore` | MySQL (`checkpoint/mysql.py`) | Production; concurrency-safe with `CheckpointExecutionBusy` / `CheckpointVersionConflict` / `CheckpointTransactionTooLarge` guards. |

Selection is driven by `PY_AGENT_CHECKPOINT_STORE` (see
`docs/configuration.md`). Both savers drop non-serializable channels
(live provider / tool bridge objects) on `put` / `put_writes`; these are
re-injected per run, so checkpoints stay msgpack-serializable.

When a checkpointer is attached, `run_workflow` invokes the graph with
`configurable.thread_id = "aca-{workflow_run_id}"` so runs are resumable.

## Tool Layer: `ToolLayer`

`tool_layer.py` defines:

- `ToolBridgeLike` — structural protocol for anything that can execute read
  tools and retrieval queries.
- `ToolLayer` — protocol with `build() -> ToolBridgeLike`.
- `GoToolBridgeLayer` — production implementation returning the configured
  `GoToolBridge` (HTTP calls to the Go backend, bearer-token authenticated).
- `StubToolLayer` — deterministic in-memory implementation for tests and the
  engineering harness.

Write operations never pass through this layer as executions; they are
returned as `ToolProposal` objects for Go-side approval (see
`docs/tool-bridge-protocol.md`).

## Assembly: `factory.py`

`factory.py` builds a fully wired harness from configuration:

- resolves the checkpoint store from `PY_AGENT_CHECKPOINT_STORE`
- resolves the tool layer (Go bridge in production, stub in tests)
- resolves the planner provider

The harness constructor accepts the protocols (`checkpoint_store:
CheckpointStore | None`, `tool_layer: ToolLayer | None`), so tests can inject
any conforming fake without patching internals. `EngineeringHarness`
(`docs/engineering-harness.md`) is built entirely from these seams.

## Tests

- `tests/test_checkpoint_store.py` — round-trip through each store via a real
  LangGraph compile/invoke.
- `tests/test_tool_layer.py` — protocol conformance for Go and stub layers.
- `tests/test_harness_factory.py` — factory wiring under different configs.
