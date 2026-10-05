"""Prometheus metrics for the Agent Runtime.

Replaces the earlier minimal in-process registry with ``prometheus-client``,
adding standard counters, gauges, and histograms for admission, workflow
duration, node duration, dependency requests, checkpoint operations, payload
bytes, retries, cancellation, and retrieval reuse.

The legacy ``registry.counter(name, description).inc()`` adapter is preserved
so existing call sites continue to work without changes.  Legacy counters
track their own value internally and are rendered alongside the standard
Prometheus metrics in the ``/metrics`` endpoint.
"""

from __future__ import annotations

import threading
from typing import Dict

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ---------------------------------------------------------------------------
# Bounded Prometheus metrics (no unbounded labels)
# ---------------------------------------------------------------------------

# Use a module-level default registry so the /metrics endpoint renders
# everything that has been registered in-process.  Tests inject their own
# ``CollectorRegistry`` to avoid cross-test pollution.
_default_registry = CollectorRegistry()

# --- Admission ---
admission_accepted_total = Counter(
    "agent_runtime_admission_accepted_total",
    "Total workflow runs admitted (acquired lease)",
    registry=_default_registry,
)
admission_rejected_total = Counter(
    "agent_runtime_admission_rejected_total",
    "Total workflow runs rejected by admission control",
    ["reason"],  # bounded: queue_full | timeout
    registry=_default_registry,
)
admission_active_runs = Gauge(
    "agent_runtime_admission_active_runs",
    "Currently active workflow runs holding a lease",
    registry=_default_registry,
)
admission_queued_runs = Gauge(
    "agent_runtime_admission_queued_runs",
    "Currently queued workflow runs waiting for a lease",
    registry=_default_registry,
)
admission_queue_wait_seconds = Histogram(
    "agent_runtime_admission_queue_wait_seconds",
    "Time spent waiting in the admission queue before acquiring a lease",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=_default_registry,
)

# --- Workflow ---
workflow_duration_seconds = Histogram(
    "agent_runtime_workflow_duration_seconds",
    "End-to-end workflow run duration",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=_default_registry,
)
workflow_runs_total = Counter(
    "agent_runtime_workflow_runs_total",
    "Total workflow runs accepted by the agent runtime",
    registry=_default_registry,
)
workflow_failures_total = Counter(
    "agent_runtime_workflow_failures_total",
    "Total workflow runs that failed",
    registry=_default_registry,
)

# --- Node ---
node_duration_seconds = Histogram(
    "agent_runtime_node_duration_seconds",
    "Duration of individual graph nodes",
    ["node"],  # bounded: known small set of node names
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=_default_registry,
)

# --- Dependency requests ---
dependency_request_total = Counter(
    "agent_runtime_dependency_request_total",
    "Total outbound dependency (provider/RAG/tool-bridge) requests",
    ["dependency"],  # bounded: provider | rag | tool_bridge
    registry=_default_registry,
)
dependency_request_errors_total = Counter(
    "agent_runtime_dependency_request_errors_total",
    "Total failed outbound dependency requests",
    ["dependency"],
    registry=_default_registry,
)
dependency_request_duration_seconds = Histogram(
    "agent_runtime_dependency_request_duration_seconds",
    "Duration of outbound dependency requests",
    ["dependency"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=_default_registry,
)

# --- Checkpoint ---
checkpoint_operation_total = Counter(
    "agent_runtime_checkpoint_operation_total",
    "Total checkpoint store operations",
    ["operation"],  # bounded: get | put | list
    registry=_default_registry,
)
checkpoint_operation_errors_total = Counter(
    "agent_runtime_checkpoint_operation_errors_total",
    "Total failed checkpoint store operations",
    ["operation"],
    registry=_default_registry,
)

# --- Payload ---
payload_bytes = Histogram(
    "agent_runtime_payload_bytes",
    "Size of workflow request/response payloads in bytes",
    ["direction"],  # bounded: request | response
    buckets=[100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000],
    registry=_default_registry,
)

# --- Retries ---
retry_total = Counter(
    "agent_runtime_retry_total",
    "Total retry attempts for dependency calls",
    ["dependency"],
    registry=_default_registry,
)

# --- Cancellation ---
cancellation_total = Counter(
    "agent_runtime_cancellation_total",
    "Total workflow runs cancelled (HTTP caller timed out)",
    registry=_default_registry,
)

# --- Retrieval reuse ---
retrieval_reuse_total = Counter(
    "agent_runtime_retrieval_reuse_total",
    "Total retrieval results reused from cache",
    registry=_default_registry,
)


# ---------------------------------------------------------------------------
# Legacy compatibility adapter
# ---------------------------------------------------------------------------

class _Counter:
    """Compatibility adapter preserving ``registry.counter(name, desc).inc()``.

    Tracks its own value internally without registering a duplicate Prometheus
    counter.  The standard Prometheus metrics defined above are the
    authoritative source for Prometheus exposition; legacy counters are
    rendered separately in ``MetricsRegistry.render_prometheus()``.
    """

    __slots__ = ("name", "description", "_lock", "_value")

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._lock = threading.Lock()
        self._value = 0

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def value(self) -> int:
        with self._lock:
            return self._value


class MetricsRegistry:
    """Legacy metrics registry with Prometheus exposition.

    Preserves the ``registry.counter(name, description).inc()`` API while
    also making all standard Prometheus metrics available via
    ``generate_latest()``.
    """

    def __init__(self, *, prom_registry: CollectorRegistry | None = None) -> None:
        self._counters: Dict[str, _Counter] = {}
        self._lock = threading.Lock()
        self._prom_registry = prom_registry or _default_registry

    def counter(self, name: str, description: str = "") -> _Counter:
        with self._lock:
            existing = self._counters.get(name)
            if existing is None:
                existing = _Counter(name, description)
                self._counters[name] = existing
            return existing

    def render_prometheus(self) -> str:
        # Render standard Prometheus metrics
        parts = [generate_latest(self._prom_registry).decode("utf-8")]
        # Render legacy counters that don't have a standard Prometheus counterpart
        with self._lock:
            items = list(self._counters.values())
        for counter in items:
            # Skip legacy counters that overlap with standard metrics
            # (they are already tracked via the standard Counter objects)
            if counter.name.startswith("agent_runtime_"):
                continue
            if counter.description:
                parts.append(f"# HELP {counter.name} {counter.description}\n")
                parts.append(f"# TYPE {counter.name} counter\n")
            parts.append(f"{counter.name} {counter.value()}\n")
        return "".join(parts)


# Module-level singleton for legacy call sites.
registry = MetricsRegistry()


def get_default_prometheus_registry() -> CollectorRegistry:
    """Return the module-level ``CollectorRegistry`` used by all metrics."""
    return _default_registry
