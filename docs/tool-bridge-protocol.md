# Tool Bridge Protocol

The Tool Bridge lets Python request authorized read-only operations from the Go backend.

## Read Tool Endpoint

`POST /api/v1/internal/agent/tools/read`

Headers:

- `Authorization: Bearer <AGENT_RUNTIME_TOOL_TOKEN>`
- `Content-Type: application/json`

Payload:

```json
{
  "organization_id": 1,
  "user_id": 7,
  "tool_name": "query_context_chunks",
  "arguments": {
    "conversation_id": 42,
    "query": "launch risk",
    "limit": 8
  }
}
```

The Go backend owns organization isolation, user permission checks, tool policy, and audit records.

## Retrieval Bridge Endpoint

`POST /api/v1/internal/agent/retrieval/query`

The RAG runtime uses this endpoint to retrieve authorized chunks before rerank and grounding.

Python services must treat bridge failure as a runtime error in strict mode and must not fall back to direct database access.

