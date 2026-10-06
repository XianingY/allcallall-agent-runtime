"""Bounded parallel execution for independent read-only agent roles."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Sequence, TypedDict, cast

from pydantic import BaseModel

from ..config import config as app_config
from ..deadline import ExecutionCancelled
from ..metrics import (
    parallel_role_cancellations_total,
    parallel_role_group_duration_seconds,
    parallel_role_groups_total,
    parallel_role_partial_failures_total,
)
from ..models import Citation, RoleResult, TraceEvent
from ..state import GraphState
from .role_router import CANONICAL_ROLE_ORDER
from .synthesis import memory_agent, searcher


RoleExecutor = Callable[[GraphState], GraphState]


class RoleDelta(TypedDict):
    """A branch-local result that has not touched shared graph state."""

    role: str
    trace_events: list[TraceEvent]
    role_result: RoleResult
    citations: list[Citation]
    action_items: list[str]
    risk_flags: list[str]


# The only initially independent pair. Synthesis depends on both roles, and the
# risk analyst depends on synthesized output, so neither can join this group.
PARALLEL_ELIGIBLE_ROLES: frozenset[str] = frozenset({"searcher", "memory_agent"})
PARALLEL_MAX_ROLES = 2
PARALLEL_PROVIDER_CALL_BUDGET = 2

ROLE_EXECUTORS: dict[str, RoleExecutor] = {
    "searcher": searcher,
    "memory_agent": memory_agent,
}

_SNAPSHOT_REFERENCE_KEYS = (
    "provider",
    "tool_bridge",
    "rag_runtime",
    "prompt_version",
)
_SNAPSHOT_LIST_KEYS = (
    "retrieved_context_chunks",
    "reranked_context_chunks",
    "agentic_context_chunks",
    "long_term_memory",
)
_SNAPSHOT_MODEL_KEYS = ("context_sufficiency", "evidence_pack")


@dataclass(slots=True)
class GroupReservation:
    """Thread-safe reservation of one bounded role group's resources."""

    roles: tuple[str, ...]
    token_budget: int
    provider_calls: int
    used_tokens: int = 0
    used_provider_calls: int = 0
    _role_tokens: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _released: bool = False

    def acquire(self, role: str, estimated_tokens: int) -> None:
        """Acquire one role's token and provider-call share."""
        with self._lock:
            if self._released:
                raise RuntimeError("group reservation has already been released")
            if role not in self.roles:
                raise ValueError(f"role {role!r} is not part of this group reservation")
            if role in self._role_tokens:
                raise RuntimeError(f"role {role!r} is already reserved")
            if estimated_tokens < 0:
                raise ValueError("estimated_tokens must not be negative")
            if self.used_tokens + estimated_tokens > self.token_budget:
                raise GroupBudgetExceeded(
                    f"token budget {self.token_budget} cannot reserve {role!r}"
                )
            if self.used_provider_calls + 1 > self.provider_calls:
                raise GroupBudgetExceeded(
                    f"provider-call budget {self.provider_calls} cannot reserve {role!r}"
                )
            self.used_tokens += estimated_tokens
            self.used_provider_calls += 1
            self._role_tokens[role] = estimated_tokens

    def release_role(self, role: str) -> None:
        """Release one role's token and provider-call share."""
        with self._lock:
            if self._released or role not in self._role_tokens:
                return
            self.used_tokens -= self._role_tokens.pop(role)
            self.used_provider_calls -= 1

    def release(self) -> None:
        """Release the whole group reservation exactly once."""
        with self._lock:
            if self._released:
                return
            self._released = True
            self.used_tokens = 0
            self.used_provider_calls = 0
            self._role_tokens.clear()


class GroupBudgetExceeded(RuntimeError):
    """Raised when a bounded role group cannot reserve enough capacity."""


def estimate_role_tokens(state: GraphState, role: str) -> int:
    """Estimate the bounded token cost of one read-only role branch."""
    request = state.get("request")
    tokens = 64 if role == "searcher" else 32
    if request is not None:
        tokens += max(1, len(request.goal) // 4)
        tokens += sum(max(1, len(chunk.snippet) // 4) for chunk in request.context_chunks[:20])
    return tokens


def _reserve_group_budget(roles: Sequence[str], token_budget: int) -> GroupReservation:
    """Reserve provider-call and token capacity for a parallel group."""
    return GroupReservation(
        roles=tuple(roles),
        token_budget=token_budget,
        provider_calls=min(len(roles), PARALLEL_PROVIDER_CALL_BUDGET),
    )


def _copy_snapshot_list(value: list[Any]) -> list[Any]:
    """Deep-copy Pydantic evidence models while preserving immutable strings."""
    return [item.model_copy(deep=True) if isinstance(item, BaseModel) else item for item in value]


def _branch_snapshot(
    state: GraphState,
    *,
    branch_cancel_event: threading.Event,
) -> GraphState:
    """Build an isolated input view for one branch.

    The request and context-bearing lists are copied. Runtime clients are safe
    to share (they own bounded process-level pools), but mutable result lists,
    trace lists, caches, and checkpoint-related objects are deliberately absent.
    """
    snapshot: dict[str, Any] = {
        "trace_events": [],
        "role_results": [],
        "branch_cancel_event": branch_cancel_event,
    }
    request = state.get("request")
    if request is not None:
        snapshot["request"] = request.model_copy(deep=True)
    snapshot.update({key: state.get(key) for key in _SNAPSHOT_REFERENCE_KEYS if key in state})
    snapshot.update(
        {
            key: _copy_snapshot_list(cast(list[Any], state.get(key)))
            for key in _SNAPSHOT_LIST_KEYS
            if key in state
        }
    )
    for key in _SNAPSHOT_MODEL_KEYS:
        value = state.get(key)
        if value is not None:
            snapshot[key] = cast(Any, value).model_copy(deep=True)
    if "skill_instructions" in state:
        snapshot["skill_instructions"] = state["skill_instructions"]
    return cast(GraphState, snapshot)


def _extract_role_result(output: dict[str, Any], role: str) -> RoleResult:
    direct = output.get(role)
    if isinstance(direct, RoleResult):
        return direct
    for result in output.get("role_results", []):
        if isinstance(result, RoleResult) and result.role == role:
            return result
    raise ValueError(f"role executor for {role} did not return a RoleResult")


def _run_role(role: str, state: GraphState, cancel_event: threading.Event) -> RoleDelta:
    branch = _branch_snapshot(state, branch_cancel_event=cancel_event)
    output = ROLE_EXECUTORS[role](branch)
    if cancel_event.is_set():
        raise ExecutionCancelled("shutdown")
    result = _extract_role_result(cast(dict[str, Any], output), role)
    return RoleDelta(
        role=role,
        trace_events=list(output.get("trace_events", [])),
        role_result=result,
        citations=list(result.citations),
        action_items=list(result.action_items),
        risk_flags=list(result.risk_flags),
    )


def merge_role_deltas(
    deltas: Sequence[RoleDelta],
    canonical_order: Sequence[str],
    *,
    parent_trace_events: Sequence[TraceEvent] | None = None,
    parent_role_results: Sequence[RoleResult] | None = None,
) -> dict[str, Any]:
    """Merge branch-local deltas into deterministic graph-state updates."""
    by_role = {delta["role"]: delta for delta in deltas}
    ordered_roles = [role for role in canonical_order if role in by_role]

    parent_trace = list(parent_trace_events or [])
    parent_results = list(parent_role_results or [])
    trace_events: list[TraceEvent] = parent_trace
    role_results: list[RoleResult] = [
        result for result in parent_results if result.role not in by_role
    ]
    citations: list[Citation] = []
    action_items: list[str] = []
    risk_flags: list[str] = []
    merged: dict[str, Any] = {}

    for role in ordered_roles:
        delta = by_role[role]
        result = delta["role_result"]
        trace_events.extend(delta["trace_events"])
        role_results.append(result)
        citations.extend(delta["citations"])
        action_items.extend(delta["action_items"])
        risk_flags.extend(delta["risk_flags"])
        merged[role] = result

    citation_keys: set[tuple[str, str, str]] = set()
    deduped_citations: list[Citation] = []
    for citation in citations:
        key = (citation.chunk_id, citation.source_type, citation.source_id)
        if key not in citation_keys:
            citation_keys.add(key)
            deduped_citations.append(citation)

    merged.update(
        {
            "trace_events": trace_events,
            "role_results": role_results,
            "citations": deduped_citations,
            "action_items": list(dict.fromkeys(action_items)),
            "risk_flags": list(dict.fromkeys(risk_flags)),
        }
    )
    return merged


def _cancel_group(futures: dict[str, Future[RoleDelta]], cancel_event: threading.Event) -> None:
    cancel_event.set()
    for future in futures.values():
        future.cancel()
    if futures:
        wait(futures.values(), timeout=app_config.cancellation_grace_seconds)


def execute_parallel_roles(
    state: GraphState,
    roles: Sequence[str],
    *,
    max_parallel: int,
    token_budget: int,
) -> list[RoleDelta]:
    """Execute the eligible independent roles with at most two branches.

    A token/provider budget shortfall forces the same roles to run sequentially.
    On cancellation or any branch failure, outstanding siblings are cancelled,
    all deltas are discarded, the reservation is released once, and the original
    exception propagates for the existing retry/deadline classification.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    unsupported = set(roles) - PARALLEL_ELIGIBLE_ROLES
    if unsupported:
        raise ValueError(
            "only searcher and memory_agent can execute in bounded parallel groups"
        )
    if not roles:
        return []

    ordered_roles = [role for role in CANONICAL_ROLE_ORDER if role in set(roles)]
    estimated_tokens = sum(estimate_role_tokens(state, role) for role in ordered_roles)
    provider_calls = len(ordered_roles)
    budget_allows_parallel = (
        len(ordered_roles) > 1
        and estimated_tokens <= token_budget
        and provider_calls <= PARALLEL_PROVIDER_CALL_BUDGET
    )

    started = time.monotonic()
    cancel_event = threading.Event()
    reservation: GroupReservation | None = None
    deltas: list[RoleDelta] = []

    try:
        reservation = _reserve_group_budget(tuple(ordered_roles), token_budget)
        if budget_allows_parallel:
            worker_count = min(max_parallel, PARALLEL_MAX_ROLES, len(ordered_roles))
            for role in ordered_roles:
                reservation.acquire(role, estimate_role_tokens(state, role))
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="role-branch",
            )
            try:
                futures = {
                    role: executor.submit(_run_role, role, state, cancel_event)
                    for role in ordered_roles
                }
                try:
                    for future in as_completed(futures.values()):
                        delta = future.result()
                        reservation.release_role(delta["role"])
                        deltas.append(delta)
                except BaseException as exc:
                    failed_role = next(
                        (
                            role
                            for role, future in futures.items()
                            if future.done() and future.exception() is not None
                        ),
                        ordered_roles[0],
                    )
                    _cancel_group(futures, cancel_event)
                    deltas.clear()
                    if isinstance(exc, ExecutionCancelled):
                        reason = exc.reason
                        parallel_role_cancellations_total.labels(reason=reason).inc()
                        parallel_role_groups_total.labels(mode="parallel", outcome="cancelled").inc()
                    else:
                        parallel_role_partial_failures_total.labels(role=failed_role).inc()
                        parallel_role_groups_total.labels(mode="parallel", outcome="partial_failure").inc()
                    raise
            finally:
                # The context manager would call shutdown(wait=True) after the
                # bounded cancellation wait and block again on an uncooperative
                # branch. Bounded shutdown leaves only that still-running worker.
                executor.shutdown(wait=False, cancel_futures=True)
            deltas.sort(key=lambda delta: CANONICAL_ROLE_ORDER.index(delta["role"]))
            parallel_role_groups_total.labels(mode="parallel", outcome="completed").inc()
        else:
            for role in ordered_roles:
                reservation.acquire(role, estimate_role_tokens(state, role))
                try:
                    delta = _run_role(role, state, cancel_event)
                    reservation.release_role(role)
                    deltas.append(delta)
                except BaseException as exc:
                    cancel_event.set()
                    deltas.clear()
                    if isinstance(exc, ExecutionCancelled):
                        reason = exc.reason
                        parallel_role_cancellations_total.labels(reason=reason).inc()
                        parallel_role_groups_total.labels(mode="sequential", outcome="cancelled").inc()
                    else:
                        parallel_role_partial_failures_total.labels(role=role).inc()
                        parallel_role_groups_total.labels(mode="sequential", outcome="partial_failure").inc()
                    raise
            parallel_role_groups_total.labels(
                mode="sequential",
                outcome="budget_sequential" if len(ordered_roles) > 1 else "completed",
            ).inc()
        return deltas
    finally:
        if reservation is not None:
            reservation.release()
        parallel_role_group_duration_seconds.observe(time.monotonic() - started)


def parallel_roles_node(state: GraphState) -> GraphState:
    """Merge a bounded searcher/memory_agent group into its parent graph state."""
    allocation = state.get("role_allocation")
    if allocation is None:
        raise ValueError("parallel_roles requires a role allocation")
    roles = [role for role in allocation.roles if role in PARALLEL_ELIGIBLE_ROLES]
    deltas = execute_parallel_roles(
        state,
        roles,
        max_parallel=PARALLEL_MAX_ROLES,
        token_budget=app_config.parallel_role_token_budget,
    )
    return cast(
        GraphState,
        merge_role_deltas(
            deltas,
            CANONICAL_ROLE_ORDER,
            parent_trace_events=state.get("trace_events", []),
            parent_role_results=state.get("role_results", []),
        ),
    )
