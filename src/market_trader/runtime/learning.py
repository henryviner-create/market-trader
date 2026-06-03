"""Close the loop: log each cycle's predictions, then grade them against outcomes.

The cycle logs every ranked name — a bounded rank-percentile "probability" plus
the feature snapshot — so the call can be replayed and scored once the forward
return is known. ``grade_predictions`` reads them back, scores calibration
(Brier) and hit-rate, computes each signal's information coefficient (IC), and
flags signals whose IC has decayed to noise. That is what lets the system learn
from its own track record instead of silently going stale.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from market_trader.backtest.metrics import brier_score
from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE
from market_trader.feedback.prediction_log import (
    PredictionRecord,
    load_predictions,
    log_predictions,
)
from market_trader.feedback.pruning import prune_signals_by_ic
from market_trader.storage.bitemporal import BitemporalStore


def log_cycle_predictions(
    store: BitemporalStore,
    scores: pd.Series,
    matrix: pd.DataFrame,
    as_of: datetime,
    *,
    model_version: str,
    horizon_days: int = 5,
) -> int:
    """Persist each scored name's rank-percentile + feature snapshot for later grading."""
    ranked = scores.dropna()
    if ranked.empty:
        return 0
    probs = ranked.rank(pct=True)  # bounded [0, 1] and scorer-agnostic
    records = [
        PredictionRecord(
            as_of=as_of,
            symbol=str(s),
            probability=float(probs[s]),
            horizon_days=horizon_days,
            model_version=model_version,
            features={k: float(v) for k, v in matrix.loc[s].dropna().items()},
        )
        for s in ranked.index
    ]
    return log_predictions(store, records)


def _forward_returns_at(panel: pd.DataFrame, t: datetime, horizon: int) -> pd.Series:
    """Per-symbol realised return over the ``horizon`` bars strictly after ``t``."""
    rets = panel.pct_change()
    pos = int(rets.index.searchsorted(t, side="right"))
    window = rets.iloc[pos : pos + horizon]
    if window.empty:
        return pd.Series(dtype=float)
    return (1.0 + window.fillna(0.0)).prod() - 1.0


def grade_predictions(
    store: BitemporalStore,
    as_of: datetime,
    *,
    horizon_days: int = 5,
    model_version: str | None = None,
    min_abs_ic: float = 0.02,
) -> dict[str, Any]:
    """Score logged predictions whose horizon has elapsed, and flag decayed signals."""
    preds = load_predictions(store, as_of, model_version=model_version)
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))

    by_time: dict[datetime, list[PredictionRecord]] = defaultdict(list)
    for p in preds:
        by_time[p.as_of].append(p)

    probs: list[float] = []
    labels: list[float] = []
    feats: list[dict[str, float]] = []
    fwd_rets: list[float] = []
    for t, group in by_time.items():
        forward = _forward_returns_at(panel, t, horizon_days) if not panel.empty else pd.Series()
        if forward.empty:
            continue
        for p in group:
            r = forward.get(p.symbol)
            if r is None or pd.isna(r):
                continue  # outcome not yet known -> not graded
            probs.append(p.probability)
            labels.append(1.0 if float(r) > 0 else 0.0)
            feats.append(p.features)
            fwd_rets.append(float(r))

    if not probs:
        return {"n": 0, "brier": 0.0, "hit_rate": 0.0, "ic": {}, "kept": [], "pruned": []}

    pa, la = np.asarray(probs), np.asarray(labels)
    feat_df = pd.DataFrame(feats)
    ret_s = pd.Series(fwd_rets)
    ic = {
        col: float(feat_df[col].corr(ret_s))
        for col in feat_df.columns
        if feat_df[col].std(skipna=True) > 0  # skip constant signals: no IC, and avoids /0
    }
    ic = {k: v for k, v in ic.items() if pd.notna(v)}  # IC of each signal vs the outcome
    kept, pruned = prune_signals_by_ic({k: abs(v) for k, v in ic.items()}, min_abs_ic=min_abs_ic)
    return {
        "n": len(probs),
        "brier": brier_score(pa, la),
        "hit_rate": float(((pa > 0.5).astype(float) == la).mean()),
        "ic": ic,
        "kept": kept,
        "pruned": pruned,
    }
