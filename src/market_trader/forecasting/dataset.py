"""Point-in-time training-set construction.

For each decision date: features come from the **PIT feature store** (no
lookahead); the label is whether the name out-performed the cross-sectional median
over the forward horizon (realised *after* the date — legitimate for supervised
training, with purged CV preventing leakage across folds). Each sample carries
``t0`` (decision time) and ``t1`` (label-realisation time) for purging.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE
from market_trader.features.base import FeatureStore
from market_trader.features.regime import macro_regime
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe import PointInTimeUniverse

REGIME_FEATURE = "regime_risk_on"


@dataclass
class TrainingSet:
    X: pd.DataFrame
    y: pd.Series
    t0: pd.Series
    t1: pd.Series
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def _realized_returns(store: BitemporalStore, dataset: str) -> pd.DataFrame:
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=dataset))
    return panel.pct_change() if not panel.empty else panel


def _forward_return(returns: pd.DataFrame, as_of: datetime, horizon: int) -> pd.Series:
    pos = int(returns.index.searchsorted(as_of, side="right"))  # first bar strictly after as_of
    window = returns.iloc[pos : pos + horizon]
    if window.empty:
        return pd.Series(dtype=float)
    return (1.0 + window.fillna(0.0)).prod() - 1.0


def build_training_set(
    store: BitemporalStore,
    feature_store: FeatureStore,
    schedule: Sequence[datetime],
    *,
    universe: PointInTimeUniverse,
    horizon_days: int = 5,
    dataset: str = PRICE_DATASET,
) -> TrainingSet:
    returns = _realized_returns(store, dataset)
    feature_names = [*feature_store.feature_names, REGIME_FEATURE]

    records: list[dict] = []
    index: list[tuple[datetime, str]] = []
    for d in sorted(schedule):
        symbols = universe.members_on(d.date())
        matrix = feature_store.compute_matrix(d, symbols)
        if matrix.empty:
            continue
        forward = _forward_return(returns, d, horizon_days)
        valid = [s for s in matrix.index if s in forward.index and pd.notna(forward[s])]
        if len(valid) < 2:
            continue
        median = forward.loc[valid].median()
        risk_on = 1.0 if macro_regime(store, d)["risk_on"] else 0.0
        label_time = d + timedelta(days=horizon_days * 2)  # generous bound for purging
        for s in valid:
            feats = matrix.loc[s].to_dict()
            feats[REGIME_FEATURE] = risk_on
            records.append({"label": int(forward[s] > median), "t0": d, "t1": label_time, **feats})
            index.append((d, str(s)))

    frame = pd.DataFrame(records, index=pd.MultiIndex.from_tuples(index, names=["date", "symbol"]))
    return TrainingSet(
        X=frame[feature_names],
        y=frame["label"],
        t0=frame["t0"],
        t1=frame["t1"],
        feature_names=feature_names,
    )
