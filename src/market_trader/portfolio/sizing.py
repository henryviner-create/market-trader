"""Unified book sizing — the one path by which scores become a risk-managed book.

``size_book`` is the single seam the live cycle and the backtest both call, so the
sizing that runs in production is the logic the backtest validates (no train/serve
skew in construction either). It holds the tradable universe, optionally *tilts* it
toward higher-scoring names, governs the book to a volatility target (the drawdown
governor), and applies the hard risk caps.

**Why a tilt-toward-1/N and not mean-variance/Kelly optimisation.** Our own
out-of-sample tests showed the signal IC on this universe is ~0, and governed
equal-weight is the hardest book to beat — the classic DeMiguel result that
optimiser estimation error swamps its theoretical edge. So the operator here is an
*exponential tilt shrunk toward equal weight* (``w ∝ exp(k·z)``): ``tilt_strength=0``
is exactly 1/N, and a validated alpha later earns a gentle, bounded lean rather than
a noise-amplifying ``Σ⁻¹μ``. This is the deploy-the-baseline / promote-alpha-on-
evidence chassis, not an alpha in itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_trader.backtest.types import Weights
from market_trader.features.base import cross_sectional_zscore
from market_trader.portfolio.construction import ledoit_wolf_cov, volatility_target_weights
from market_trader.portfolio.risk import RiskLimits, apply_risk_limits


def _tilt_weights(scores: pd.Series | None, names: list[str], tilt_strength: float) -> pd.Series:
    """Equal weight, optionally tilted toward higher scores (``w ∝ exp(k·z)``).

    The exponential form keeps every weight positive (long-only, no clip) and makes
    ``tilt_strength`` read in cross-sectional-std units; ``tilt_strength=0`` collapses
    to 1/N. The z-score is clipped so one outlier name cannot dominate the book.
    """
    base = pd.Series(1.0 / len(names), index=names)
    if scores is None or tilt_strength <= 0:
        return base
    z = cross_sectional_zscore(scores.reindex(names).astype(float).fillna(0.0)).clip(-3.0, 3.0)
    raw = pd.Series(np.exp(tilt_strength * z.to_numpy()), index=names)
    total = float(raw.sum())
    return raw / total if total > 0 else base


def size_book(
    returns: pd.DataFrame,
    *,
    target_vol: float,
    limits: RiskLimits,
    scores: pd.Series | None = None,
    tilt_strength: float = 0.0,
    lookback: int = 90,
    sectors: dict[str, str] | None = None,
    regime_derisk: float = 1.0,
) -> Weights:
    """Turn the tradable universe (and optional scores) into a risk-managed book.

    ``returns`` is the trailing per-symbol return panel (covariance + the vol
    governor). ``scores`` is optional cross-sectional conviction; with ``tilt_strength``
    it leans the book toward higher scores, and ``tilt_strength=0`` (or ``scores is
    None``) returns the governed equal-weight baseline. ``regime_derisk`` (<= 1.0)
    shrinks the vol budget in a risk-off regime. Names without enough clean trailing
    history to risk-size are dropped (holding them would run ungoverned risk); when
    there is too little history to estimate any covariance the book degrades to the
    un-governed equal/tilt weights (a fresh deployment). Returns ``{symbol: weight}``.
    """
    names = [str(c) for c in returns.columns]
    if not names:
        return {}
    raw = _tilt_weights(scores, names, tilt_strength)
    budget = target_vol * max(0.0, regime_derisk)

    window = returns[names].tail(lookback).dropna(axis=1, how="any")
    if window.shape[1] >= 2 and window.shape[0] >= 20:
        cov = ledoit_wolf_cov(window)
        w = raw.reindex(cov.columns).fillna(0.0)
        gross = float(w.sum())
        if gross > 0:
            w = w / gross  # renormalise over the governable names before scaling
        sized = volatility_target_weights(w, cov, budget)
    else:
        sized = raw * max(0.0, regime_derisk)  # too little history -> ungoverned cold start

    book = {str(k): float(v) for k, v in sized.items() if abs(float(v)) > 1e-9}
    # apply_risk_limits enforces per-name, gross (caps a calm-market lever-up), net,
    # and (when a map is supplied) sector limits — the hard floor under the governor.
    return apply_risk_limits(book, limits, sectors=sectors)
