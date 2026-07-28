"""Hierarchical memory + context compression (Module 4).

Long conversations (meeting transcripts, multi-turn chat, retrieved notes) can
inflate the model context window and degrade both quality and cost. This module
provides:

* :func:`build_model_history` — produce a *bounded* ``modelHistory`` string
  from a list of conversation turns using one of two strategies:
    - ``"summary"`` — compress older turns into a single extractive summary and
      keep the most recent turns verbatim,
    - ``"full"`` — keep only the most recent turns that fit the token budget and
      drop older turns entirely (full replacement of the tail window).
* :class:`LongTermMemoryStore` — a pluggable long-term memory backend
  (relational today, vector DB tomorrow) decoupled from the runtime so the
  short-term GraphState never has to hold unbounded history.

Both strategies are deterministic and testable without an LLM; a real
summarizer can be injected for production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

DEFAULT_MAX_TOKENS = 4000

SUMMARY_STRATEGY = "summary"
FULL_STRATEGY = "full"

# A rough, dependency-free token estimate. CJK characters are ~1 token each;
# other scripts ~1 token per 3 characters. Good enough for budgeting.
_CJK_START, _CJK_END = ord("一"), ord("鿿")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _CJK_START <= ord(ch) <= _CJK_END)
    other = len(text) - cjk
    return max(1, cjk + other // 3)


@dataclass
class ConversationTurn:
    role: str
    content: str


Summarizer = Callable[[list[ConversationTurn]], str]


def _extractive_summary(turns: list[ConversationTurn], max_tokens: int) -> str:
    """Deterministic, LLM-free summary: keep the first and last turn verbatim
    and join them, trimming to the token budget."""
    if not turns:
        return ""
    head = turns[0]
    tail = turns[-1] if len(turns) > 1 else None
    parts = [f"[{head.role}]: {head.content}"]
    if tail is not None:
        parts.append(f"[{tail.role}]: {tail.content}")
    summary = " … ".join(parts)
    while estimate_tokens(summary) > max_tokens and len(summary) > 10:
        summary = summary[: max(10, len(summary) * 3 // 4)]
    return summary


def _compress_summary(turns: list[ConversationTurn], budget: int, summarizer: Summarizer | None) -> str:
    if summarizer is not None:
        return summarizer(turns)
    return _extractive_summary(turns, budget)


def _compress_full(turns: list[ConversationTurn], budget: int) -> str:
    """Keep the most recent turns that fit; drop older turns entirely."""
    selected: list[ConversationTurn] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn.content) + 2  # +2 for the role prefix
        if selected and used + cost > budget:
            break
        selected.insert(0, turn)
        used += cost
    return "\n".join(f"[{t.role}]: {t.content}" for t in selected)


def build_model_history(
    turns: list[ConversationTurn],
    strategy: str = FULL_STRATEGY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    summarizer: Summarizer | None = None,
) -> str:
    """Build a bounded ``modelHistory`` string from ``turns``.

    The result is guaranteed to stay within ``max_tokens`` for both strategies.
    """
    if not turns:
        return ""
    if strategy == SUMMARY_STRATEGY:
        return _compress_summary(turns, max_tokens, summarizer)
    if strategy == FULL_STRATEGY:
        return _compress_full(turns, max_tokens)
    raise ValueError(f"unknown compression strategy: {strategy!r}")


@runtime_checkable
class LongTermMemoryStore(Protocol):
    """Pluggable long-term memory backend.

    Today this is backed by the Go ``AgentMemory`` relational slots; the same
    interface can later be served by a vector DB for semantic recall without
    touching any runtime node.
    """

    def put(self, key: str, value: str, embedding: list[float] | None = None) -> None:
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        ...


@dataclass
class InMemoryLongTermMemory:
    """Default in-process long-term memory store (used in tests / single-shot)."""

    _data: dict[str, str] = field(default_factory=dict)

    def put(self, key: str, value: str, embedding: list[float] | None = None) -> None:
        del embedding
        self._data[key] = value

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        del query
        return list(self._data.values())[: max(0, top_k)]
