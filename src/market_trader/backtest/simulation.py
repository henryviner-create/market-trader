"""Monte-Carlo simulation of a strategy's return distribution.

A backtest yields one realised path; that single number hides the range of
plausible outcomes. This resamples the realised daily returns with a **block**
bootstrap (preserving short-horizon autocorrelation) into many alternative
paths, then reports the *distribution* of terminal return, worst drawdown, and
Sharpe — so performance is shown with its uncertainty, and the downside tail is
explicit. Reuses ``backtest/metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from market_trader.backtest.metrics import TRADING_DAYS, max_drawdown, sharpe_ratio


def block_bootstrap_paths(
    returns: ArrayLike, *, n_sims: int = 2000, block: int = 10, seed: int = 0
) -> NDArray[np.float64]:
    """``n_sims`` resampled return paths, each the length of ``returns``."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size == 0:
        return np.empty((0, 0), dtype=float)
    block = max(1, min(block, r.size))
    rng = np.random.default_rng(seed)
    n = r.size
    n_blocks = int(np.ceil(n / block))
    paths = np.empty((n_sims, n), dtype=float)
    for i in range(n_sims):
        starts = rng.integers(0, n, size=n_blocks)
        seq = np.concatenate([np.take(r, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
        paths[i] = seq
    return paths


@dataclass(frozen=True)
class SimulationReport:
    n_sims: int
    total_return_q05: float
    total_return_q50: float
    total_return_q95: float
    max_drawdown_q05: float  # 5th percentile = the worst-tail drawdowns
    max_drawdown_q50: float
    sharpe_q50: float
    prob_positive: float


def monte_carlo_report(
    returns: ArrayLike,
    *,
    n_sims: int = 2000,
    block: int = 10,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS,
) -> SimulationReport:
    """Distribution of terminal return / drawdown / Sharpe across resampled paths."""
    paths = block_bootstrap_paths(returns, n_sims=n_sims, block=block, seed=seed)
    if paths.size == 0:
        return SimulationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    totals = np.prod(1.0 + paths, axis=1) - 1.0
    mdds = np.array([max_drawdown(p) for p in paths])
    sharpes = np.array([sharpe_ratio(p, periods_per_year) for p in paths])
    return SimulationReport(
        n_sims=int(paths.shape[0]),
        total_return_q05=float(np.quantile(totals, 0.05)),
        total_return_q50=float(np.quantile(totals, 0.50)),
        total_return_q95=float(np.quantile(totals, 0.95)),
        max_drawdown_q05=float(np.quantile(mdds, 0.05)),
        max_drawdown_q50=float(np.quantile(mdds, 0.50)),
        sharpe_q50=float(np.quantile(sharpes, 0.50)),
        prob_positive=float(np.mean(totals > 0.0)),
    )
