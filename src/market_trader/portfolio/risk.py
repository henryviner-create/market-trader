"""Hard risk controls — the layer above execution.

Every target portfolio passes through here: per-name and gross/sector exposure
limits clip or scale weights, ``check_order`` *refuses* an over-limit order in
code (it raises), and the drawdown circuit-breaker halts trading when cumulative
losses cross a threshold. Execution is downstream and can never bypass this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_trader.backtest.types import PointInTimeView, Strategy, Weights


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.10
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_sector_weight: float = 0.30


class RiskLimitBreach(Exception):
    """Raised when an order would violate a hard limit (fail closed)."""


def apply_risk_limits(
    weights: Weights, limits: RiskLimits, *, sectors: dict[str, str] | None = None
) -> Weights:
    """Clip per-name weights, then scale to the gross (and sector) exposure caps."""
    s = pd.Series(weights, dtype=float)
    if s.empty:
        return {}
    s = s.clip(lower=-limits.max_position_weight, upper=limits.max_position_weight)

    gross = float(s.abs().sum())
    if gross > limits.max_gross_exposure and gross > 0:
        s = s * (limits.max_gross_exposure / gross)

    if sectors:
        for sector in set(sectors.values()):
            members = [k for k in s.index if sectors.get(str(k)) == sector]
            exposure = float(s[members].abs().sum())
            if exposure > limits.max_sector_weight and exposure > 0:
                s[members] = s[members] * (limits.max_sector_weight / exposure)

    # Net-exposure cap: trim the dominant side toward neutral (shrinks gross, never
    # flips a name's sign) so a long/short book can't drift to unintended directional
    # risk. This was previously unenforced even though the limit field existed.
    net = float(s.sum())
    if abs(net) > limits.max_net_exposure + 1e-12:
        dominant = s > 0 if net > 0 else s < 0
        dom_gross = float(s[dominant].abs().sum())
        if dom_gross > 0:
            s.loc[dominant] = s[dominant] * (1.0 - (abs(net) - limits.max_net_exposure) / dom_gross)

    return {str(k): float(v) for k, v in s.items() if v != 0.0}


def check_order(
    symbol: str, target_weight: float, current_weights: Weights, limits: RiskLimits
) -> None:
    """Raise :class:`RiskLimitBreach` if the order would breach a hard limit."""
    if abs(target_weight) > limits.max_position_weight + 1e-9:
        raise RiskLimitBreach(
            f"position {symbol}={target_weight:.4f} exceeds {limits.max_position_weight}"
        )
    proposed = {**current_weights, symbol: target_weight}
    gross = sum(abs(v) for v in proposed.values())
    if gross > limits.max_gross_exposure + 1e-9:
        raise RiskLimitBreach(f"gross {gross:.4f} exceeds {limits.max_gross_exposure}")
    net = sum(proposed.values())
    if abs(net) > limits.max_net_exposure + 1e-9:
        raise RiskLimitBreach(f"net {net:.4f} exceeds {limits.max_net_exposure}")


class DrawdownCircuitBreaker:
    """Trips (and stays tripped) once drawdown from the peak crosses the threshold."""

    def __init__(self, max_drawdown: float = 0.20) -> None:
        self.max_drawdown = max_drawdown
        self._peak: float | None = None
        self.tripped = False

    def update(self, equity: float) -> bool:
        self._peak = equity if self._peak is None else max(self._peak, equity)
        if self._peak > 0 and (equity / self._peak - 1.0) <= -self.max_drawdown:
            self.tripped = True
        return self.tripped

    def reset(self) -> None:
        self._peak = None
        self.tripped = False


@dataclass
class RiskManagedStrategy:
    """Wraps a strategy so its targets always pass through the risk limits."""

    base: Strategy
    limits: RiskLimits
    sectors: dict[str, str] | None = None
    name: str = "risk_managed"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        raw = self.base.target_weights(view, as_of)
        return apply_risk_limits(raw, self.limits, sectors=self.sectors)
