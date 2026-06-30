#!/usr/bin/env sh
set -eu

curl -sS http://127.0.0.1:8091/v1/retrieval/agentic \
  -H 'Content-Type: application/json' \
  -d '{
    "organization_id": 1,
    "user_id": 7,
    "conversation_id": 42,
    "query": "What launch risk was discussed?",
    "top_k": 3,
    "max_steps": 2,
    "chunks": [
      {
        "chunk_id": "mt-1",
        "source_type": "meeting_transcript",
        "source_id": "segment-101",
        "source_title": "Launch meeting transcript",
        "snippet": "The launch risk is delayed supplier approval.",
        "score": 92
      },
      {
        "chunk_id": "msg-1",
        "source_type": "message",
        "source_id": "message-12",
        "source_title": "General chat",
        "snippet": "Please update the weekly notes.",
        "score": 20
      }
    ]
  }' | python -m json.tool

