# Skill Registry

Skills are the Agent-facing abstraction over AllCallAll product capabilities.
This document covers both the capability catalog (the contract with the Go
backend) and the dynamic registry implementation (`skill_registry.py`).

## Capability Catalog

### Read Skills

- `query_context_chunks`
- `query_knowledge_chunks`
- `query_meeting_transcript_segments`
- `query_recent_followups`
- `query_recent_meetings`
- `query_conversation_members`
- `query_contact_profile`

Read skills may be requested by the Agent Runtime and executed by the Go
backend after permission checks.

### Write Skills

- `write_conversation_message`
- `create_follow_up_task`
- `upsert_conversation_memory`

Write skills are proposal-only in Python. The response must include tool
name, arguments, reason, idempotency key, and `approval_required=true`.

### Async Tool Queue Metadata

Write proposals also include queue metadata for the Go backend:

- `execution_mode=async_after_approval`
- `queue_name`
- `priority`
- `max_attempts`
- `rate_limit_key`
- `dead_letter_queue`

Python never executes these writes. Go owns approval creation, schema
validation, queue enqueue, retry, dead-letter handling, audit logs, and final
side effects. Queue semantics are modeled in
`docs/mcp-tools-async-queue.md`.

## Dynamic Registry Implementation (`skill_registry.py`)

### `Skill`

Each registered skill carries: `name`, `instructions`, `tools`, `risk_level`
(`low` | `high`), `scope`, and `instruction_sha256` — a SHA-256 snapshot of
the instruction text taken at registration time for tamper evidence. If the
loaded instructions ever diverge from the snapshot, the mismatch is
detectable.

### Loading from `SKILL.md`

`parse_skill_md` parses a `SKILL.md` file: YAML frontmatter (name, tools,
risk_level, scope) plus a Markdown instruction body. `SkillRegistry` supports:

- `register(skill)` — programmatic registration
- `load_skill_md(path)` — single-file load
- `load_directory(path)` — bulk load of a skills directory
- `get(name)` / `resolve(name)` — lookup and resolution

### `SecurityOverlay`

`resolve()` returns a `ResolvedSkill`. For `risk_level: high` skills the
`SecurityOverlay` force-applies:

- the safety plan (`SAFETY_PLAN`) prepended to the effective instructions
- `requires_approval = True`, regardless of what the skill file declares

Low-risk skills resolve unchanged. This guarantees a high-risk skill can
never opt out of the approval gate by editing its own frontmatter.

## Tests

`tests/test_skill_registry.py` (7 cases): frontmatter parsing, snapshot
stability and tamper evidence, low- vs high-risk resolution, unknown-skill
error, and file/directory loading.
