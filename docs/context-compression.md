# Hierarchical Memory and Context Compression

`context_compression.py` adds a token-bounded model-history layer on top of
the existing short-term memory. It is opt-in: with
`PY_AGENT_ENABLE_CONTEXT_COMPRESSION=false` (default) request handling is
byte-for-byte identical to the legacy path.

## Memory Layers

| Layer | Location | Persistence |
| --- | --- | --- |
| Working state | LangGraph `GraphState` | Per run (checkpointable, see `docs/harness-architecture.md`). |
| Short-term memory | harness in-memory store | Per session. |
| Compressed model history | `build_model_history` output injected into requests | Per request, token-bounded. |
| Long-term memory | `LongTermMemoryStore` protocol (+ `InMemoryLongTermMemory` reference impl) | Pluggable; production backends can implement the protocol. |

## Compression Strategies

`build_model_history(turns, max_tokens, strategy)` supports:

- `summary` (default) — extractive compression: keeps the most
  information-dense turns within the token budget.
- `full` — recency-only: keeps the most recent turns that fit the budget.

Token estimation (`estimate_tokens`) treats CJK characters as ~1 token each
and other text as ~1/3 token per character.

## Request Wiring

- `MeetingBriefRequest.model_history: str` (added in `models.py`; contracts
  schema regenerated accordingly).
- `helpers.build_request_model_history(request, ...)` converts messages,
  notes, and meeting transcripts into `ConversationTurn`s and compresses them.
- `request_with_runtime_context` injects the compressed history only when
  `PY_AGENT_ENABLE_CONTEXT_COMPRESSION=true`.

## Configuration

See `docs/configuration.md` — `PY_AGENT_ENABLE_CONTEXT_COMPRESSION`,
`PY_AGENT_CONTEXT_COMPRESSION_STRATEGY`, `PY_AGENT_MODEL_HISTORY_MAX_TOKENS`.

## Tests

`tests/test_context_compression.py` (10 cases): token estimation, budget
enforcement for both strategies, unknown-strategy error, long-term store
round-trip, and opt-in request wiring.
