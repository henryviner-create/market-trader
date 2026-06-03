"""Conditional signal weighting.

The composite score combines z-scored signals with weights that the data sets and
we constrain: ``Score(a) = Σ_i w_i · quality_i · z(signal_i(a))``. Weights come
from a ladder (equal -> inverse-variance -> IC), capped and decayed; an
orthogonality penalty down-weights redundant signals. The score is a **ranking
input** to the model/portfolio layers, never a direct trade trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_trader.features.base import cross_sectional_zscore


def information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> float:
    """Rank correlation (Spearman IC) between a signal and forward returns."""
    df = pd.concat([signal, forward_returns], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def equal_weights(names: pd.Index | list[str]) -> pd.Series:
    names = pd.Index(names)
    if len(names) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names)


def inverse_variance_weights(matrix: pd.DataFrame) -> pd.Series:
    var = matrix.var(ddof=0)
    inv = (1.0 / var.replace(0.0, np.nan)).fillna(0.0)
    total = inv.sum()
    return equal_weights(matrix.columns) if total == 0 else inv / total


def ic_weights(ics: pd.Series, *, cap: float = 0.5) -> pd.Series:
    """Sign-aware weights from per-signal IC, normalised by total |IC| and capped."""
    denom = ics.abs().sum()
    if denom == 0:
        return equal_weights(ics.index)
    return (ics / denom).clip(lower=-cap, upper=cap)


def orthogonality_penalty(matrix: pd.DataFrame) -> pd.Series:
    """``1 / (1 + Σ_j≠i |corr(i, j)|)`` — redundant signals are down-weighted."""
    corr = matrix.corr().abs().fillna(0.0)
    diagonal = pd.Series(np.diag(corr.to_numpy()), index=corr.index)
    redundancy = corr.sum(axis=1) - diagonal  # exclude self-correlation, no mutation
    return 1.0 / (1.0 + redundancy)


def composite_score(
    matrix: pd.DataFrame,
    weights: pd.Series,
    *,
    quality: pd.Series | None = None,
    decay: pd.Series | None = None,
) -> pd.Series:
    """Cross-sectional composite score per symbol (signed weights orient signals)."""
    z = matrix.apply(cross_sectional_zscore, axis=0)
    w = weights.reindex(matrix.columns).fillna(0.0)
    if quality is not None:
        w = w * quality.reindex(matrix.columns).fillna(1.0)
    if decay is not None:
        w = w * decay.reindex(matrix.columns).fillna(1.0)
    scale = w.abs().sum()
    if scale > 0:
        w = w / scale
    return z.mul(w, axis=1).sum(axis=1)
