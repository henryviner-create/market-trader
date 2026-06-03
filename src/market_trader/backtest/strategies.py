"""Baseline and toy strategies.

The baselines exist to be beaten (or not): every candidate is judged net of costs
against equal-weight and buy-and-hold. :class:`MomentumStrategy` is a placeholder
candidate to exercise the harness — not a claim of edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_trader.backtest.types import PointInTimeView, Weights


@dataclass
class EqualWeightStrategy:
    name: str = "equal_weight"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        universe = view.universe()
        if not universe:
            return {}
        w = 1.0 / len(universe)
        return {s: w for s in universe}


@dataclass
class MomentumStrategy:
    lookback: int = 20
    top_fraction: float = 0.5
    name: str = "momentum"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        panel = view.price_panel()
        if panel.empty or panel.shape[0] <= self.lookback:
            return EqualWeightStrategy().target_weights(view, as_of)
        window = panel.ffill().iloc[-(self.lookback + 1) :]
        momentum = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
        if momentum.empty:
            return {}
        k = max(1, int(len(momentum) * self.top_fraction))
        winners = [str(s) for s in momentum.sort_values(ascending=False).head(k).index]
        w = 1.0 / len(winners)
        return {s: w for s in winners}
