# Architecture

This repository is the Python intelligence layer for AllCallAll.

```text
AllCallAll Go Backend
  auth / org isolation / DB / approvals / audit / write execution
      |
      | HTTP JSON + bearer token
      v
Python Agent Runtime
  LangGraph workflows / bounded ReAct / prompt registry / traces / proposals
      |
      v
Python RAG Runtime
  retrieval planning / rerank / evidence pack / grounding checks
```

The runtime does not directly access AllCallAll MySQL, Redis, Elasticsearch, or object storage. Production data access goes through Go-owned read tools and retrieval bridge endpoints.

## Main Workflows

- `react_general`: natural-language task handling with bounded read-tool loops.
- `meeting_brief`: grounded meeting recap from transcript and retrieved context.
- `risk_review`: risk extraction with citations and conservative missing-context behavior.
- `follow_up_planner`: action item generation as approval-required tool proposals.
- `context_qa`: grounded Q&A over conversation, transcript, and knowledge context.

## Safety Model

- Read tools can be called automatically through the Go Tool Bridge.
- Write tools are never executed by Python.
- Python returns `ToolProposal` objects with `approval_required=true`.
- Go validates schema, creates approvals, audits decisions, and performs final writes.

