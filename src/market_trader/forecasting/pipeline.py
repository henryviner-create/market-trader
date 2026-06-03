"""Training / evaluation pipeline and the engine-facing forecast strategy.

Honest evaluation only: ``purged_cv_auc`` scores with purged k-fold + embargo;
``time_split`` carves a disjoint out-of-sample test period; ``ForecastStrategy``
recomputes PIT features at each rebalance so the backtest replays through
knowledge time exactly as live would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from market_trader.backtest.metrics import brier_score, reliability_curve
from market_trader.backtest.splitters import PurgedKFold
from market_trader.backtest.types import PointInTimeView, Weights
from market_trader.features.base import FeatureStore
from market_trader.features.regime import macro_regime
from market_trader.forecasting.dataset import REGIME_FEATURE, TrainingSet, build_training_set
from market_trader.forecasting.models import Forecaster
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe import PointInTimeUniverse


@dataclass
class CalibrationReport:
    n: int
    auc: float
    brier: float
    reliability: pd.DataFrame


def evaluate_predictions(probs: Any, labels: Any, n_bins: int = 10) -> CalibrationReport:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan")
    return CalibrationReport(
        n=int(p.size),
        auc=auc,
        brier=brier_score(p, y),
        reliability=reliability_curve(p, y, n_bins),
    )


def time_split(schedule: Sequence[datetime], train_frac: float = 0.6) -> tuple[list, list]:
    ordered = sorted(schedule)
    k = int(len(ordered) * train_frac)
    return ordered[:k], ordered[k:]


def train_forecaster(
    store: BitemporalStore,
    feature_store: FeatureStore,
    train_dates: Sequence[datetime],
    *,
    universe: PointInTimeUniverse,
    make_model: Callable[[], Any],
    horizon_days: int = 5,
) -> tuple[Forecaster, TrainingSet]:
    ts = build_training_set(
        store, feature_store, train_dates, universe=universe, horizon_days=horizon_days
    )
    model = Forecaster(make_model(), "forecast").fit(ts.X.to_numpy(dtype=float), ts.y.to_numpy())
    return model, ts


def purged_cv_auc(
    ts: TrainingSet, make_model: Callable[[], Any], *, n_splits: int = 5, embargo: float = 0.02
) -> float:
    """Mean test AUC over purged k-fold splits (embargoed). NaN if not scorable."""
    X = ts.X.to_numpy(dtype=float)
    y = ts.y.to_numpy()
    t0 = ts.t0.to_numpy()
    t1 = ts.t1.to_numpy()
    aucs: list[float] = []
    for train_idx, test_idx in PurgedKFold(n_splits=n_splits, embargo=embargo).split(t0, t1):
        if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
            continue
        model = Forecaster(make_model(), "cv").fit(X[train_idx], y[train_idx])
        aucs.append(float(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx]))))
    return float(np.mean(aucs)) if aucs else float("nan")


@dataclass
class ForecastStrategy:
    store: BitemporalStore
    feature_store: FeatureStore
    model: Forecaster
    universe: PointInTimeUniverse
    feature_names: list[str]
    top_quantile: float = 0.3
    name: str = "forecast"
    _regime_cache: dict[datetime, float] = field(default_factory=dict, repr=False)

    def target_weights(self, view: PointInTimeView, as_of: datetime) -> Weights:
        symbols = self.universe.members_on(as_of.date())
        matrix = self.feature_store.compute_matrix(as_of, symbols)
        if matrix.empty:
            return {}
        features = matrix.copy()
        features[REGIME_FEATURE] = 1.0 if macro_regime(self.store, as_of)["risk_on"] else 0.0
        probs = self.model.predict_proba(features[self.feature_names].to_numpy(dtype=float))
        ranked = pd.Series(probs, index=matrix.index).dropna().sort_values(ascending=False)
        if ranked.empty:
            return {}
        k = max(1, int(len(ranked) * self.top_quantile))
        winners = list(ranked.head(k).index)
        weight = 1.0 / len(winners)
        return {str(sym): weight for sym in winners}
