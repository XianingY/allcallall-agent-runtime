# Skill Registry

Skills are the Agent-facing abstraction over AllCallAll product capabilities.

## Read Skills

- `query_context_chunks`
- `query_knowledge_chunks`
- `query_meeting_transcript_segments`
- `query_recent_followups`
- `query_recent_meetings`
- `query_conversation_members`
- `query_contact_profile`

Read skills may be requested by the Agent Runtime and executed by the Go backend after permission checks.

## Write Skills

- `write_conversation_message`
- `create_follow_up_task`
- `upsert_conversation_memory`

Write skills are proposal-only in Python. The response must include tool name, arguments, reason, idempotency key, and `approval_required=true`.

## Async Tool Queue Metadata

Write proposals also include queue metadata for the Go backend:

- `execution_mode=async_after_approval`
- `queue_name`
- `priority`
- `max_attempts`
- `rate_limit_key`
- `dead_letter_queue`

Python never executes these writes. Go owns approval creation, schema validation, queue enqueue, retry, dead-letter handling, audit logs, and final side effects.
