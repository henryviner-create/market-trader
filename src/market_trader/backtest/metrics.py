"""Performance and calibration metrics.

We optimise for risk-adjusted return, so the harness reports Sharpe / Sortino /
Calmar / max-drawdown / turnover / hit-rate — and, because a forecaster that says
"70%" should be right ~70% of the time, **calibration** (reliability curve +
Brier score) as a first-class citizen. Bootstrap confidence intervals are
provided so no number is ever shown without its uncertainty.

All ratios degrade gracefully to ``0.0`` on degenerate inputs (too few points,
zero variance) rather than returning ``inf``/``nan`` that would poison reports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

TRADING_DAYS = 252


def _clean(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float).ravel()
    return a[~np.isnan(a)]


def annualized_return(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    if total <= 0.0:
        return -1.0
    return total ** (periods_per_year / r.size) - 1.0


def annualized_vol(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: int = TRADING_DAYS, rf: float = 0.0
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - rf / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike, periods_per_year: int = TRADING_DAYS, rf: float = 0.0
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - rf / periods_per_year
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0.0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    """Most negative peak-to-trough of the compounded equity curve (<= 0)."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def calmar_ratio(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def hit_rate(returns: ArrayLike) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0.0))


def brier_score(probs: ArrayLike, outcomes: ArrayLike) -> float:
    p = np.asarray(probs, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if p.size == 0:
        return 0.0
    return float(np.mean((p - o) ** 2))


def reliability_curve(probs: ArrayLike, outcomes: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Binned predicted-probability vs. realised-frequency table.

    A well-calibrated forecaster has ``mean_pred ≈ frac_pos`` in every populated bin.
    """
    p = np.asarray(probs, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_pred": float(p[mask].mean()) if count else float("nan"),
                "frac_pos": float(o[mask].mean()) if count else float("nan"),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    returns: ArrayLike,
    statistic: Callable[..., float] = sharpe_ratio,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 0,
) -> tuple[float, float]:
    """``(low, high)`` percentile bootstrap CI for a return statistic."""
    r = _clean(returns)
    if r.size < 2:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(r, size=r.size, replace=True)
        stats[i] = statistic(sample, periods_per_year)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


@dataclass(frozen=True)
class PerformanceSummary:
    n_periods: int
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hit_rate: float
    avg_turnover: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def summarize(
    returns: ArrayLike,
    turnover: Sequence[float] | ArrayLike | None = None,
    periods_per_year: int = TRADING_DAYS,
    rf: float = 0.0,
) -> PerformanceSummary:
    r = _clean(returns)
    avg_turnover = 0.0
    if turnover is not None:
        t = _clean(turnover)
        avg_turnover = float(t.mean()) if t.size else 0.0
    return PerformanceSummary(
        n_periods=int(r.size),
        ann_return=annualized_return(r, periods_per_year),
        ann_vol=annualized_vol(r, periods_per_year),
        sharpe=sharpe_ratio(r, periods_per_year, rf),
        sortino=sortino_ratio(r, periods_per_year, rf),
        calmar=calmar_ratio(r, periods_per_year),
        max_drawdown=max_drawdown(r),
        hit_rate=hit_rate(r),
        avg_turnover=avg_turnover,
    )
