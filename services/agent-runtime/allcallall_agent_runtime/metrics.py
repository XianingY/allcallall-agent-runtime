from __future__ import annotations

import threading
from typing import Dict


class _Counter:
    __slots__ = ("name", "description", "_value", "_lock")

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def value(self) -> int:
        with self._lock:
            return self._value


class MetricsRegistry:
    """Minimal, dependency-free in-process metrics registry.

    Exposes Prometheus text exposition without pulling in prometheus-client so
    the runtime stays lean. Counters are process-local and reset on restart;
    a real deployment should scrape ``/metrics`` and/or ship to OTLP.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, _Counter] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> _Counter:
        with self._lock:
            existing = self._counters.get(name)
            if existing is None:
                existing = _Counter(name, description)
                self._counters[name] = existing
            return existing

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            items = list(self._counters.values())
        for counter in items:
            if counter.description:
                lines.append(f"# HELP {counter.name} {counter.description}")
                lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.value()}")
        return "\n".join(lines) + "\n"


registry = MetricsRegistry()
