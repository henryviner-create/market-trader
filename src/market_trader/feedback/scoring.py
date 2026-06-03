"""Score logged predictions against realised outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np

from market_trader.backtest.metrics import brier_score
from market_trader.feedback.prediction_log import PredictionRecord


def score_predictions(
    predictions: list[PredictionRecord],
    realized_returns: dict[str, float],
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Brier + hit-rate over predictions whose outcome (forward return) is known."""
    probs: list[float] = []
    labels: list[float] = []
    for p in predictions:
        outcome = realized_returns.get(p.symbol)
        if outcome is None:
            continue
        probs.append(p.probability)
        labels.append(1.0 if outcome > threshold else 0.0)

    n = len(probs)
    if n == 0:
        return {"n": 0, "brier": 0.0, "hit_rate": 0.0}
    pa = np.asarray(probs, dtype=float)
    la = np.asarray(labels, dtype=float)
    hit_rate = float(((pa > 0.5).astype(float) == la).mean())
    return {"n": n, "brier": brier_score(pa, la), "hit_rate": hit_rate}
