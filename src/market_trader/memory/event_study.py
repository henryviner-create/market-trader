"""Event-study impact engine.

Estimate a counterfactual "normal" return with a market model
(``r_i = alpha + beta * r_market``) over a pre-event estimation window, then the
**Abnormal Return** (AR) over the event window and the **Cumulative Abnormal
Return** (CAR). Aggregate CARs across events of a type into the full outcome
*distribution* (mean, dispersion, t-stat, tails) — never a point average.

The anchor is whatever date the caller passes. For a *predictive* (tradable)
study, anchor on the **knowledge time** and use ``pre=0`` so the window is the
abnormal return earned *after* the event became knowable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class MarketModel:
    alpha: float
    beta: float
    resid_std: float


def estimate_market_model(stock: NDArray[np.float64], market: NDArray[np.float64]) -> MarketModel:
    n = len(stock)
    design = np.column_stack([np.ones(n), market])
    coef, *_ = np.linalg.lstsq(design, stock, rcond=None)
    resid = stock - design @ coef
    dof = max(n - 2, 1)
    return MarketModel(float(coef[0]), float(coef[1]), float(np.sqrt(resid @ resid / dof)))


def event_car(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    anchor: datetime,
    *,
    estimation_days: int = 60,
    gap_days: int = 5,
    pre: int = 0,
    post: int = 5,
) -> float | None:
    """Cumulative abnormal return around ``anchor``, or None if data is insufficient."""
    df = pd.concat([stock_returns.rename("s"), market_returns.rename("m")], axis=1).dropna()
    if df.empty:
        return None
    pos = int(df.index.searchsorted(anchor, side="right")) - 1  # last bar at/before anchor
    if pos < 0:
        return None
    est_end = pos - gap_days
    est_start = est_end - estimation_days
    if est_start < 0 or est_end <= est_start or pos + post >= len(df):
        return None

    est = df.iloc[est_start:est_end]
    model = estimate_market_model(est["s"].to_numpy(dtype=float), est["m"].to_numpy(dtype=float))
    window = df.iloc[pos - pre : pos + post + 1]
    abnormal = window["s"].to_numpy(dtype=float) - (
        model.alpha + model.beta * window["m"].to_numpy(dtype=float)
    )
    return float(abnormal.sum())


@dataclass(frozen=True)
class EventOutcomeDistribution:
    label: str
    n: int
    mean_car: float
    std_car: float
    t_stat: float
    share_positive: float
    q10: float
    q50: float
    q90: float

    def significant(self, threshold: float = 1.96) -> bool:
        return self.n > 1 and abs(self.t_stat) >= threshold


def aggregate_event_study(
    events: Iterable[tuple[str, datetime]],
    returns_panel: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    label: str = "event",
    estimation_days: int = 60,
    gap_days: int = 5,
    pre: int = 0,
    post: int = 5,
) -> EventOutcomeDistribution:
    """Run ``event_car`` for each (entity, anchor) event and aggregate the CARs."""
    market = market_returns if market_returns is not None else returns_panel.mean(axis=1)
    cars: list[float] = []
    for entity_id, anchor in events:
        if entity_id not in returns_panel.columns:
            continue
        car = event_car(
            returns_panel[entity_id],
            market,
            anchor,
            estimation_days=estimation_days,
            gap_days=gap_days,
            pre=pre,
            post=post,
        )
        if car is not None and not np.isnan(car):
            cars.append(car)

    arr: NDArray[np.float64] = np.array(cars, dtype=float)
    n = int(arr.size)
    if n == 0:
        return EventOutcomeDistribution(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mean / (std / np.sqrt(n))) if std > 0 else 0.0
    return EventOutcomeDistribution(
        label=label,
        n=n,
        mean_car=mean,
        std_car=std,
        t_stat=t_stat,
        share_positive=float((arr > 0).mean()),
        q10=float(np.quantile(arr, 0.1)),
        q50=float(np.quantile(arr, 0.5)),
        q90=float(np.quantile(arr, 0.9)),
    )


def returns_from_events(events: Sequence[tuple[str, datetime]]) -> list[str]:
    """Convenience: the distinct entities referenced by a set of events."""
    return sorted({e for e, _ in events})
