"""Tests for the prometheus-backed metrics module."""

from __future__ import annotations

import threading

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from allcallall_agent_runtime.metrics import (
    MetricsRegistry,
    _Counter,
    get_default_prometheus_registry,
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


class TestStandardMetricsIsolated:
    """Verify standard Prometheus metric types work correctly using isolated registries."""

    def test_counter_with_labels(self) -> None:
        reg = CollectorRegistry()
        c = Counter("test_labeled_counter", "test", ["reason"], registry=reg)
        c.labels(reason="queue_full").inc()
        c.labels(reason="timeout").inc(2)
        samples = list(c.collect())
        assert len(samples) == 1
        total_samples = [s for s in samples[0].samples if not s.name.endswith("_created")]
        label_values = {s.labels["reason"]: s.value for s in total_samples if "reason" in s.labels}
        assert label_values.get("queue_full") == 1.0
        assert label_values.get("timeout") == 2.0

    def test_gauge_set_and_dec(self) -> None:
        reg = CollectorRegistry()
        g = Gauge("test_gauge", "test gauge", registry=reg)
        g.set(5)
        g.inc(3)
        output = generate_latest_text(reg)
        assert "test_gauge 8.0" in output

    def test_histogram_observations(self) -> None:
        reg = CollectorRegistry()
        h = Histogram("test_histogram", "test hist", buckets=[1, 5, 10], registry=reg)
        h.observe(0.5)
        h.observe(3.0)
        h.observe(7.0)
        output = generate_latest_text(reg)
        assert "test_histogram_count 3" in output
        assert "test_histogram_sum" in output

    def test_counter_without_labels(self) -> None:
        reg = CollectorRegistry()
        c = Counter("test_simple_counter", "test", registry=reg)
        c.inc(10)
        output = generate_latest_text(reg)
        assert "test_simple_counter_total 10.0" in output

    def test_default_registry_accessible(self) -> None:
        reg = get_default_prometheus_registry()
        assert reg is not None


def generate_latest_text(registry: CollectorRegistry) -> str:
    """Helper to generate Prometheus text output from a registry."""
    from prometheus_client import generate_latest
    return generate_latest(registry).decode("utf-8")
