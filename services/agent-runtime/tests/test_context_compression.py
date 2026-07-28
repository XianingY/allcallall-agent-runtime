"""Tests for hierarchical memory + context compression (Module 4)."""

from __future__ import annotations

import pytest

from allcallall_agent_runtime.config import config as app_config
from allcallall_agent_runtime.context_compression import (
    FULL_STRATEGY,
    SUMMARY_STRATEGY,
    ConversationTurn,
    InMemoryLongTermMemory,
    build_model_history,
    estimate_tokens,
)


def _turns(n: int, size: int = 200) -> list[ConversationTurn]:
    return [ConversationTurn(role="user" if i % 2 == 0 else "assistant",
                             content=f"turn {i} " + "x" * size) for i in range(n)]


def test_estimate_tokens_nonzero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 1


def test_full_strategy_keeps_recent_within_budget() -> None:
    turns = _turns(20, size=200)
    # Each turn ~ (200/3 + 2) tokens; budget 300 keeps only the last couple.
    out = build_model_history(turns, strategy=FULL_STRATEGY, max_tokens=300)
    assert estimate_tokens(out) <= 300
    # Most recent turn must survive.
    assert "turn 19" in out
    # Oldest turns are dropped under full replacement.
    assert "turn 0" not in out


def test_summary_strategy_bounded_and_deterministic() -> None:
    turns = _turns(20, size=200)
    out = build_model_history(turns, strategy=SUMMARY_STRATEGY, max_tokens=400)
    assert estimate_tokens(out) <= 400
    # Extractive summary keeps head + tail.
    assert "turn 0" in out
    assert "turn 19" in out


def test_summary_strategy_accepts_injected_summarizer() -> None:
    turns = _turns(5, size=50)
    out = build_model_history(
        turns, strategy=SUMMARY_STRATEGY, max_tokens=400,
        summarizer=lambda t: "SUMMARY:" + str(len(t)),
    )
    assert out == "SUMMARY:5"


def test_empty_turns_yield_empty_history() -> None:
    assert build_model_history([]) == ""


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        build_model_history(_turns(3), strategy="bogus")


def test_in_memory_long_term_store_put_retrieve() -> None:
    store = InMemoryLongTermMemory()
    store.put("m1", "meeting about Q3 roadmap")
    store.put("m2", "follow-up: assign owner to risk register")
    got = store.retrieve("roadmap", top_k=5)
    assert "meeting about Q3 roadmap" in got
    assert "follow-up: assign owner to risk register" in got
    assert len(store.retrieve("x", top_k=1)) == 1


def test_build_request_model_history_is_bounded() -> None:
    from allcallall_agent_runtime.helpers import build_request_model_history
    from allcallall_agent_runtime.models import ConversationMessage, MeetingBriefRequest

    request = MeetingBriefRequest(
        organization_id=1, user_id=1, conversation_id=1, workflow_run_id=1, goal="g",
        messages=[ConversationMessage(sender_id=i, body=f"long message number {i} " + "y" * 300)
                  for i in range(30)],
    )
    history = build_request_model_history(request, strategy=FULL_STRATEGY, max_tokens=400)
    assert history  # non-empty
    assert estimate_tokens(history) <= 400
    # Bounded: not all 30 messages survive the full-replacement budget.
    assert history.count("long message number") < 30


def test_request_context_injects_model_history_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from allcallall_agent_runtime import helpers
    from allcallall_agent_runtime.models import ConversationMessage, MeetingBriefRequest

    monkeypatch.setattr(app_config, "enable_context_compression", True)
    request = MeetingBriefRequest(
        organization_id=1, user_id=1, conversation_id=1, workflow_run_id=1, goal="g",
        messages=[ConversationMessage(sender_id=i, body=f"msg {i} " + "z" * 200) for i in range(10)],
    )
    state = {"request": request.model_dump(), "reranked_context_chunks": []}
    out = helpers.request_with_runtime_context(state)
    assert out.model_history  # injected and bounded
    assert estimate_tokens(out.model_history) <= app_config.model_history_max_tokens


def test_request_context_omits_model_history_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from allcallall_agent_runtime import helpers
    from allcallall_agent_runtime.models import ConversationMessage, MeetingBriefRequest

    monkeypatch.setattr(app_config, "enable_context_compression", False)
    request = MeetingBriefRequest(
        organization_id=1, user_id=1, conversation_id=1, workflow_run_id=1, goal="g",
        messages=[ConversationMessage(sender_id=i, body=f"msg {i}") for i in range(10)],
    )
    state = {"request": request.model_dump(), "reranked_context_chunks": []}
    out = helpers.request_with_runtime_context(state)
    assert out.model_history == ""  # legacy behavior preserved by default
