# MCP Tool Wrapping and Async Tool Queue

Module implementation for exposing runtime capabilities as MCP-shaped tool
descriptors and modeling the approval-gated asynchronous execution queue.
Files: `mcp_tools.py`, `async_tool_queue.py`.

## MCP Tool Registry (`mcp_tools.py`)

`MCPTool` describes a tool in MCP list-tools shape:

- `name`, `title`, `description`, `input_schema`
- `kind`: `READ` or `WRITE`
- `execution_mode`: `EXEC_SYNC` (reads) or `EXEC_ASYNC` (writes, only after
  Go-side approval)
- `to_mcp()` renders the standard MCP tool dict.

`MCPToolRegistry` holds descriptors; `default_registry()` registers the
canonical set:

| Tool | Kind | Execution mode |
| --- | --- | --- |
| `query_context` | READ | sync |
| `markdown_write` | WRITE | async after approval |
| `meeting_transcribe` | WRITE | async after approval |

The read/write split mirrors the safety boundary: reads may execute through
the Tool Bridge, writes are proposal-only (`docs/tool-bridge-protocol.md`).

## Async Tool Queue (`async_tool_queue.py`)

`AsyncToolQueue` models the queue semantics the Go backend applies to
approved write proposals. It is deterministic (caller-supplied clock) and
backed by the `TaskStore` protocol (`InMemoryTaskStore` reference impl).

### Semantics

- Idempotent enqueue — a duplicate `idempotency_key` returns the existing
  task instead of creating a new one.
- Lease-based claim — `claim(worker)` leases a task for
  `visibility_timeout`; a crashed worker's task becomes claimable again after
  the lease expires (tasks in `processing` with an expired lease are eligible).
- Per-key rate limiting — at most N in-flight tasks per `rate_limit_key`.
- Retry with exponential backoff — `fail()` reschedules at
  `base_backoff * 2^(attempts-1)` until `max_attempts`.
- Dead-letter queue — exhausted tasks move to `dead` status and are listed
  via `dead_letters()`.
- Priority ordering — higher-priority tasks are claimed first.

### Task lifecycle

```text
queued --claim--> processing --complete--> done
   ^                  |
   |            fail (attempts < max, backoff)
   +------------------+
                      |
                fail (attempts == max)
                      v
                    dead  (DLQ)
```

## Tests

`tests/test_mcp_async_queue.py` (9 cases): tool classification and MCP list
shape, read/write split, idempotent enqueue, lease + complete, expired-lease
reclaim, rate-limit throttling, retry/backoff/DLQ, priority ordering.
