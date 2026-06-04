"""Measure a signal's out-of-sample information coefficient (IC) over history.

Replays each feature point-in-time over past decision dates and correlates it with
the *forward* return realised after that date — the standard cross-sectional rank
IC, averaged over dates, with a t-stat. This is the gate that says whether a signal
(e.g. the newly-wired insider-buying feature) actually predicts returns, without
waiting weeks for live predictions to mature.

No lookahead: features are computed as of each date (knowledge_time <= date) and the
forward return uses only the fully-elapsed horizon after that date.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.features.base import Feature
from market_trader.runtime.learning import _forward_returns_at  # matured-horizon forward returns
from market_trader.storage.bitemporal import BitemporalStore


@dataclass(frozen=True)
class SignalIC:
    signal: str
    n_dates: int  # decision dates that contributed an IC
    n_obs: int  # total (name, date) pairs scored
    mean_ic: float  # mean per-date cross-sectional rank IC
    ic_std: float  # std of the per-date ICs
    ic_t_stat: float  # mean_ic / (ic_std / sqrt(n_dates)) — significance
    hit_rate: float  # fraction of dates with IC > 0


def _summarize(signal: str, ics: list[float], n_obs: int) -> SignalIC:
    arr = np.asarray(ics, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t = float(mean / (sd / np.sqrt(n)))
    elif n > 1 and mean != 0.0:
        t = (
            float("inf") if mean > 0 else float("-inf")
        )  # zero variance, nonzero mean: fully consistent
    else:
        t = 0.0
    return SignalIC(signal, n, int(n_obs), mean, sd, t, float((arr > 0).mean()))


def measure_signal_ic(
    store: BitemporalStore,
    features: Sequence[Feature],
    symbols: Sequence[str],
    as_of: datetime,
    *,
    horizon_days: int = 5,
    every: int = 5,
    warmup: int = 60,
    max_dates: int = 120,
    min_names: int = 5,
) -> dict[str, SignalIC]:
    """Per-signal cross-sectional rank IC vs forward returns, averaged over dates.

    Returns a mapping ``signal -> SignalIC``; signals with no usable variation on a
    date are skipped that date, and signals with no usable dates are omitted. Note: if
    ``every < horizon_days`` the forward windows overlap, which inflates the t-stat —
    sample at ``every >= horizon_days`` for independent (honest) significance.
    """
    panel = observations_to_price_frame(store.as_of(as_of, dataset=PRICE_DATASET))
    if panel.empty:
        return {}
    dates = [d.to_pydatetime() for d in pd.DatetimeIndex(panel.index)][warmup::every][-max_dates:]
    syms = list(symbols)
    per_date: dict[str, list[float]] = defaultdict(list)
    n_obs: dict[str, int] = defaultdict(int)
    for d in dates:
        fwd = _forward_returns_at(panel, d, horizon_days)
        if fwd.empty:
            continue
        # Price-derived features compute from a cheap in-memory slice of the panel
        # (no per-date DB query or re-pivot); other features read their own dataset.
        price_slice = panel.loc[panel.index <= d]
        cols: dict[str, pd.Series] = {}
        for f in features:
            from_panel = getattr(f, "_from_panel", None)
            cols[f.name] = (
                from_panel(price_slice, syms)
                if from_panel is not None
                else f.compute(store, d, syms)
            )
        matrix = pd.DataFrame(cols)
        if matrix.empty:
            continue
        common = [s for s in matrix.index if s in fwd.index and pd.notna(fwd[s])]
        if len(common) < min_names:
            continue
        ranked_fwd = fwd.loc[common].rank()
        for col in matrix.columns:
            vals = matrix.loc[common, col]
            sd = vals.std(skipna=True)
            if int(vals.notna().sum()) < min_names or pd.isna(sd) or sd == 0:
                continue
            ic = float(vals.rank().corr(ranked_fwd))  # Spearman = Pearson of ranks
            if not pd.isna(ic):
                per_date[str(col)].append(ic)
                n_obs[str(col)] += len(common)
    return {col: _summarize(col, ics, n_obs[col]) for col, ics in per_date.items() if ics}
