"""Transaction-cost models.

Costs are not optional decoration: a backtest without them reports an upper
bound that live trading will never reach. The default model charges commission,
half-spread, and slippage against one-way turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from market_trader.backtest.types import Weights


def one_way_turnover(prev: Weights, new: Weights) -> float:
    """Sum of absolute weight changes, ``sum|w_new - w_prev|``.

    A full switch out of one name and into another contributes 2.0 (you trade
    both legs), which is what the cost model is charged on.
    """
    symbols = set(prev) | set(new)
    return float(sum(abs(new.get(s, 0.0) - prev.get(s, 0.0)) for s in symbols))


@runtime_checkable
class CostModel(Protocol):
    def turnover_cost(self, prev: Weights, new: Weights) -> float: ...


@dataclass(frozen=True)
class BasicCostModel:
    """Linear cost in basis points of one-way turnover."""

    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0

    def turnover_cost(self, prev: Weights, new: Weights) -> float:
        bps = self.commission_bps + self.half_spread_bps + self.slippage_bps
        return one_way_turnover(prev, new) * bps * 1e-4


@dataclass(frozen=True)
class ZeroCostModel:
    """Frictionless costs — for isolating signal from costs in tests only."""

    def turnover_cost(self, prev: Weights, new: Weights) -> float:
        return 0.0
