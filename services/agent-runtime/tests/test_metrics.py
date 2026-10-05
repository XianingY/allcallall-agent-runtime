"""Tests for the prometheus-backed metrics module."""

from __future__ import annotations

import threading

from prometheus_client import CollectorRegistry

from allcallall_agent_runtime.metrics import (
    MetricsRegistry,
    _Counter,
    admission_accepted_total,
    admission_active_runs,
    admission_queued_runs,
    admission_queue_wait_seconds,
    admission_rejected_total,
    cancellation_total,
    checkpoint_operation_errors_total,
    checkpoint_operation_total,
    dependency_request_duration_seconds,
    dependency_request_errors_total,
    dependency_request_total,
    get_default_prometheus_registry,
    node_duration_seconds,
    payload_bytes,
    retrieval_reuse_total,
    retry_total,
    workflow_duration_seconds,
    workflow_failures_total,
    workflow_runs_total,
)


class TestCounterCompat:
    """Legacy ``registry.counter(name, desc).inc()`` adapter."""

    def test_counter_inc_increments_value(self) -> None:
        c = _Counter("test_c", "test counter")
        assert c.value() == 0
        c.inc()
        assert c.value() == 1
        c.inc(3)
        assert c.value() == 4

    def test_counter_is_thread_safe(self) -> None:
        c = _Counter("test_thread_c", "threaded counter")
        n = 1000
        threads = [threading.Thread(target=lambda: c.inc()) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.value() == n


class TestMetricsRegistry:
    """MetricsRegistry preserves the legacy API."""

    def test_counter_returns_same_object_for_same_name(self) -> None:
        reg = MetricsRegistry()
        c1 = reg.counter("my_counter", "desc")
        c2 = reg.counter("my_counter", "desc")
        assert c1 is c2

    def test_counter_inc_via_registry(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("reg_counter", "via registry")
        c.inc()
        assert c.value() == 1

    def test_render_prometheus_produces_text(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("rendered_counter", "rendered desc")
        c.inc(7)
        output = reg.render_prometheus()
        assert "rendered_counter" in output
        assert "rendered desc" in output

    def test_render_prometheus_includes_standard_metrics(self) -> None:
        reg = MetricsRegistry()
        output = reg.render_prometheus()
        # Standard metrics should be present in the output
        assert "agent_runtime_workflow_runs_total" in output


class TestStandardMetrics:
    """Verify that all standard Prometheus metrics are importable and usable."""

    def test_admission_metrics_exist(self) -> None:
        assert admission_accepted_total is not None
        assert admission_rejected_total is not None
        assert admission_active_runs is not None
        assert admission_queued_runs is not None
        assert admission_queue_wait_seconds is not None

    def test_workflow_metrics_exist(self) -> None:
        assert workflow_duration_seconds is not None
        assert workflow_runs_total is not None
        assert workflow_failures_total is not None

    def test_node_metrics_exist(self) -> None:
        assert node_duration_seconds is not None

    def test_dependency_metrics_exist(self) -> None:
        assert dependency_request_total is not None
        assert dependency_request_errors_total is not None
        assert dependency_request_duration_seconds is not None

    def test_checkpoint_metrics_exist(self) -> None:
        assert checkpoint_operation_total is not None
        assert checkpoint_operation_errors_total is not None

    def test_payload_metrics_exist(self) -> None:
        assert payload_bytes is not None

    def test_retry_metrics_exist(self) -> None:
        assert retry_total is not None

    def test_cancellation_metrics_exist(self) -> None:
        assert cancellation_total is not None

    def test_retrieval_metrics_exist(self) -> None:
        assert retrieval_reuse_total is not None

    def test_default_registry_accessible(self) -> None:
        reg = get_default_prometheus_registry()
        assert reg is not None

    def test_labeled_metrics_accept_labels(self) -> None:
        reg = CollectorRegistry()
        from prometheus_client import Counter
        c = Counter("test_labeled", "test", ["reason"], registry=reg)
        c.labels(reason="queue_full").inc()
        c.labels(reason="timeout").inc(2)
        samples = list(c.collect())
        assert len(samples) == 1
        # Filter to actual counter samples (not _created/_bucket etc.)
        total_samples = [s for s in samples[0].samples if not s.name.endswith("_created")]
        label_values = {s.labels["reason"]: s.value for s in total_samples if "reason" in s.labels}
        assert label_values.get("queue_full") == 1.0
        assert label_values.get("timeout") == 2.0
