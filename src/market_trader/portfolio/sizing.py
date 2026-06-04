"""Unified risk-based sizing — the drawdown governor in one place.

Composes the (previously unused) construction toolkit into a single pipeline so the
live cycle and the backtest can size positions identically:

    scores -> fractional-Kelly tilt (relative conviction)
           -> volatility target (absolute scale to a vol budget, regime-adjusted)
           -> hard risk limits (per-name / gross / NET / sector caps)

The volatility target is how the drawdown cap is enforced in practice: a ~25% DD
budget maps to roughly a 10% annualized vol target (halved for fat tails). Kelly is
used only for the *relative* tilt from conviction (the cross-sectional score), never
as a literal return forecast — hence the heavy fractional haircut.
"""

from __future__ import annotations

import pandas as pd

from market_trader.backtest.types import Weights
from market_trader.portfolio.construction import (
    fractional_kelly_weights,
    ledoit_wolf_cov,
    volatility_target_weights,
)
from market_trader.portfolio.risk import RiskLimits, apply_risk_limits


def size_book(
    scores: pd.Series,
    returns: pd.DataFrame,
    *,
    target_vol: float,
    limits: RiskLimits,
    kelly_fraction: float = 0.25,
    sectors: dict[str, str] | None = None,
    regime_derisk: float = 1.0,
) -> Weights:
    """Turn signed cross-sectional ``scores`` into risk-managed target weights.

    ``scores`` is signed conviction (positive = long, negative = short); ``returns``
    is the trailing per-symbol return panel used to estimate covariance. ``regime_derisk``
    (<= 1.0) shrinks the vol target in risk-off regimes. Returns ``{symbol: weight}``;
    empty if there is nothing to size.
    """
    names = [str(s) for s in scores.index if s in returns.columns]
    if not names:
        return {}
    mu = scores.reindex(names).astype(float).fillna(0.0)
    if float(mu.abs().sum()) == 0.0:
        return {}

    cov = ledoit_wolf_cov(returns[names])
    # Relative conviction tilt (Kelly on the score, not on return units).
    raw = fractional_kelly_weights(mu, cov, fraction=kelly_fraction)
    if float(raw.abs().sum()) == 0.0:
        raw = mu  # degenerate covariance -> fall back to the raw score directions
    # Absolute scale to the (regime-adjusted) volatility budget.
    sized = volatility_target_weights(raw, cov, target_vol * max(0.0, regime_derisk))
    # Hard caps (per-name / gross / net / sector).
    return apply_risk_limits({str(k): float(v) for k, v in sized.items()}, limits, sectors=sectors)
