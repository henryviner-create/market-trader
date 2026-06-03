"""Drift monitoring: alert when live performance diverges from the backtest."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftMonitor:
    """Compares a live metric to its backtest baseline within a tolerance band."""

    baseline: float
    tolerance: float = 0.10
    metric: str = "hit_rate"

    def is_drifting(self, live_value: float) -> bool:
        return abs(live_value - self.baseline) > self.tolerance

    def message(self, live_value: float) -> str:
        gap = live_value - self.baseline
        return (
            f"{self.metric} drift: live={live_value:.3f} vs backtest={self.baseline:.3f} "
            f"(gap {gap:+.3f}, tolerance ±{self.tolerance:.3f})"
        )
