"""Minimal Prometheus-format metrics (no client dependency).

Thread-safe counters and gauges rendered in the Prometheus text exposition
format, served at ``/metrics`` by the engine. Kept in-house so the engine image
stays lean; swap for ``prometheus_client`` later if richer types are needed.
"""

from __future__ import annotations

import threading


class Counter:
    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help = help_text
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help = help_text
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    @property
    def value(self) -> float:
        return self._value


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, tuple[str, Counter | Gauge]] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str = "") -> Counter:
        return self._get_or_create(name, help_text, "counter")  # type: ignore[return-value]

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        return self._get_or_create(name, help_text, "gauge")  # type: ignore[return-value]

    def _get_or_create(self, name: str, help_text: str, kind: str) -> Counter | Gauge:
        with self._lock:
            if name not in self._metrics:
                metric: Counter | Gauge = (
                    Counter(name, help_text) if kind == "counter" else Gauge(name, help_text)
                )
                self._metrics[name] = (kind, metric)
            return self._metrics[name][1]

    def render(self) -> str:
        lines: list[str] = []
        for name, (kind, metric) in sorted(self._metrics.items()):
            if metric.help:
                lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name} {metric.value}")
        return "\n".join(lines) + "\n"


_DEFAULT = MetricsRegistry()


def default_registry() -> MetricsRegistry:
    return _DEFAULT
