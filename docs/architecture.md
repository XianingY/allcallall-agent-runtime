# Architecture

This repository is the Python intelligence layer for AllCallAll.

```text
AllCallAll Go Backend
  auth / org isolation / DB / approvals / audit / write execution
      |
      | HTTP JSON + bearer token
      v
Python Agent Runtime
  LangGraph workflows / bounded ReAct / dynamic routing / memory reflection
      |
      v
Python RAG Runtime
  adaptive RAG / graph expansion / rerank / evidence pack / grounding checks
```

The runtime does not directly access AllCallAll MySQL, Redis, Elasticsearch, or object storage. Production data access goes through Go-owned read tools and retrieval bridge endpoints.

## Main Workflows

- `react_general`: natural-language task handling with bounded read-tool loops.
- `meeting_brief`: grounded meeting recap from transcript and retrieved context.
- `risk_review`: risk extraction with citations and conservative missing-context behavior.
- `follow_up_planner`: action item generation as approval-required tool proposals.
- `context_qa`: grounded Q&A over conversation, transcript, and knowledge context.

## Agent Runtime Harness

Each run follows a bounded, auditable harness:

1. Collect request context, modality metadata, permissions, and prompt version.
2. Route intent into `chat`, `consult`, or `risk`.
3. Build an adaptive retrieval plan with source scopes and optional graph-expanded terms.
4. Run Searcher, MemoryAgent, Summarizer, and RiskGuardian-style roles with trace events.
5. Merge role outputs, run grounding checks, and produce reflection memory.
6. Return approval-required write proposals with async queue metadata.

The graph keeps deterministic rule fallbacks so local evals do not require live LLM, vector DB, or Go backend access.

## Adaptive RAG

- `chat`: starts from scoped conversation, message, note, follow-up, and memory context.
- `consult`: knowledge-first retrieval with graph-expanded policy/checklist terms.
- `risk`: multi-hop retrieval over transcript, policy, conversation, and memory evidence.
- Evidence packs include selected chunk IDs, source coverage, citations, context sufficiency, and inferred knowledge-graph edges.

## Safety Model

- Read tools can be called automatically through the Go Tool Bridge.
- Write tools are never executed by Python.
- Python returns `ToolProposal` objects with `approval_required=true`.
- Tool proposals include idempotency keys, async queue names, retry limits, rate-limit keys, and dead-letter queues.
- Go validates schema, creates approvals, audits decisions, enqueues accepted write tools, and performs final writes.
