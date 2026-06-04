"""Baseline and toy strategies.

The baselines exist to be beaten (or not): every candidate is judged net of costs
against equal-weight and buy-and-hold. :class:`MomentumStrategy` is a placeholder
candidate to exercise the harness — not a claim of edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_trader.backtest.types import PointInTimeView, Strategy, Weights


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


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


@dataclass
class CompositeBacktestStrategy:
    """Price z-score composite of momentum / mean-reversion / volatility (top-N,
    inverse-vol weighted) — the same shape as the live cycle.

    When ``insider_scores`` (a point-in-time ``{rebalance -> (symbol -> net buys)}``
    map) is supplied, the validated insider signal joins as an equal 4th z-scored
    component, so a backtest can A/B the price-only book against the insider-tilted
    one net of costs. Absent it, behaviour is the original price-only composite.
    """

    momentum_lookback: int = 60
    meanrev_lookback: int = 5
    vol_window: int = 20
    max_positions: int = 20
    top_quantile: float = 0.3
    insider_scores: dict[datetime, pd.Series] | None = None
    name: str = "composite"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        panel = view.price_panel().ffill()
        if panel.shape[0] <= self.momentum_lookback + 1:
            return EqualWeightStrategy().target_weights(view, as_of)
        rets = panel.pct_change()
        feat = pd.DataFrame(
            {
                "mom": panel.iloc[-1] / panel.iloc[-(self.momentum_lookback + 1)] - 1.0,
                "meanrev": -(panel.iloc[-1] / panel.iloc[-(self.meanrev_lookback + 1)] - 1.0),
                "vol": rets.iloc[-self.vol_window :].std(),
            }
        ).dropna()
        if feat.empty:
            return {}
        zscores = feat[["mom", "meanrev", "vol"]].apply(_zscore, axis=0)
        # Blend the validated insider signal as an equal 4th component when this
        # rebalance's scores were precomputed point-in-time; absent that, the composite
        # is the original price-only one (an all-zero insider column can't reorder it).
        scores = None if self.insider_scores is None else self.insider_scores.get(as_of)
        if scores is not None:
            zscores = zscores.assign(insider=_zscore(scores.reindex(feat.index).fillna(0.0)))
        composite = zscores.mean(axis=1)
        k = max(1, min(self.max_positions, int(len(composite) * self.top_quantile)))
        winners = list(composite.sort_values(ascending=False).head(k).index)

        vols = feat["vol"].reindex(winners)
        inv = (1.0 / vols).where(vols > 0)
        if inv.notna().any():
            inv = inv.fillna(inv.mean())
            total = float(inv.sum())
            if total > 0:
                return {str(s): float(inv.loc[s] / total) for s in winners}
        w = 1.0 / len(winners)
        return {str(s): w for s in winners}


@dataclass
class VolTargetedStrategy:
    """Scale any strategy's book to a target annualised volatility — the DD governor.

    Wraps an inner strategy: takes its weights, estimates the held names' covariance
    (Ledoit-Wolf) over a trailing window from the point-in-time view, and rescales so the
    book's annualised volatility equals ``target_vol`` — capped at ``max_gross`` so a calm
    market can't lever it without bound. Cutting exposure as volatility rises is what keeps
    realised drawdown inside the governor; on a long-only book it mostly trades into cash.
    """

    inner: Strategy
    target_vol: float = 0.10
    max_gross: float = 1.0
    lookback: int = 90
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}@{self.target_vol:.0%}vol"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        from market_trader.portfolio.construction import (
            ledoit_wolf_cov,
            volatility_target_weights,
        )

        weights = self.inner.target_weights(view, as_of)
        if not weights:
            return weights
        rets = view.returns_panel()
        held = [s for s in weights if s in rets.columns]
        window = rets[held].tail(self.lookback).dropna(axis=1, how="any")
        if window.shape[1] < 2 or window.shape[0] < 20:
            return weights  # too little history to estimate covariance — leave unscaled
        cov = ledoit_wolf_cov(window)
        w = pd.Series(weights).reindex(cov.columns).fillna(0.0)
        scaled = volatility_target_weights(w, cov, self.target_vol)
        gross = float(scaled.abs().sum())
        if gross > self.max_gross and gross > 0:
            scaled = scaled * (self.max_gross / gross)
        return {str(s): float(v) for s, v in scaled.items() if abs(float(v)) > 1e-9}


@dataclass
class LongShortInsiderStrategy:
    """Dollar-neutral cross-sectional book on the insider signal.

    Long the strongest net-buying names, short the strongest net-selling, equal weight
    within each leg at ``gross / 2`` a side — so the book is ~market-neutral and the
    validated insider rank-edge stands alone, stripped of the market beta that caps a
    long-only book at roughly passive's Sharpe. ``insider_scores`` is the same
    point-in-time ``{rebalance -> (symbol -> net buys)}`` map used elsewhere; only names
    with actual disclosed activity *and* a tradable price enter the book.
    """

    insider_scores: dict[datetime, pd.Series]
    max_positions_per_side: int = 10
    gross: float = 1.0
    name: str = "ls_insider"

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        scores = self.insider_scores.get(as_of)
        if scores is None:
            return {}
        tradable = set(view.universe())
        active = scores.dropna()
        active = active[(active != 0.0) & active.index.isin(tradable)]
        ranked = active.sort_values()
        k = min(self.max_positions_per_side, ranked.shape[0] // 2)
        if k < 1:
            return {}
        per = self.gross / (2.0 * k)
        out = {str(s): -per for s in ranked.index[:k]}  # most net selling -> short
        out.update({str(s): per for s in ranked.index[-k:]})  # most net buying -> long
        return out
