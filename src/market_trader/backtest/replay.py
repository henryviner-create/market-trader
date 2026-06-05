"""Walk-forward replay of the *learning* system on point-in-time history.

Unlike ``simulate`` (a fixed strategy over a pre-built panel), this replays the engine as it
would have run live, **advancing the learning loop step by step**: at each rebalance it
grades the predictions whose horizon has elapsed, re-weights the composite by that *measured*
IC, scores and sizes the book through the same ``size_book`` chassis, and logs fresh
predictions for the next grading. It is how we "train the system as if it were N years ago"
on the quant side — and it yields two things at once: a simulated, honest (point-in-time)
track record, and the **live-IC time series**, which is the strongest overfit detector (the
IC the system actually measured on itself over history should match the backtest IC).

PIT integrity is the load-bearing detail: the store is built **incrementally** (prices are
revealed only up to each step), so the grader can never score an immature prediction against
a price from its future — the one subtlety that separates a faithful replay from a
look-ahead-contaminated one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_trader.backtest.costs import BasicCostModel, CostModel
from market_trader.backtest.metrics import TRADING_DAYS, PerformanceSummary, summarize
from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.schema import Observation
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.features.base import Feature, FeatureStore
from market_trader.portfolio.risk import RiskLimits
from market_trader.portfolio.sizing import size_book
from market_trader.runtime.learning import grade_predictions, log_cycle_predictions
from market_trader.runtime.scoring import composite_scorer, ic_weighted_scorer
from market_trader.storage import InMemoryBitemporalStore


@dataclass
class ReplayResult:
    equity_curve: pd.Series
    net_returns: pd.Series
    ic_history: pd.DataFrame  # per-rebalance measured IC of each signal (the live learning track)
    summary: PerformanceSummary


def replay_learning(
    observations: Sequence[Observation],
    *,
    universe: Sequence[str],
    schedule: Sequence[datetime],
    features: Sequence[Feature],
    horizon_days: int = 21,
    target_vol: float = 0.10,
    limits: RiskLimits | None = None,
    tilt_strength: float = 1.0,
    min_abs_ic: float = 0.02,
    cost_model: CostModel | None = None,
    model_version: str = "replay",
) -> ReplayResult:
    """Replay the IC-learning book over ``schedule`` on point-in-time ``observations``.

    ``observations`` is the full history (prices, and any other datasets the ``features``
    read); it is revealed to the system only up to each rebalance. The composite is
    re-weighted each step by the IC the learning loop measured on already-graded predictions,
    then ``size_book`` builds the book (a ``tilt_strength``-strength lean on that score).
    Returns the net-of-cost equity curve plus the measured-IC history.
    """
    limits = limits or RiskLimits()
    cost_model = cost_model or BasicCostModel()
    names = {str(s) for s in universe}
    rebalances = sorted(schedule)

    price_obs = [o for o in observations if o.dataset == PRICE_DATASET]
    full = observations_to_price_frame(price_obs)
    full = full[[c for c in full.columns if str(c) in names]]
    if full.empty or len(rebalances) < 2:
        raise ValueError("replay needs price data and at least two rebalance dates")
    full_returns = full.pct_change()

    # Incremental store: reveal each observation only once its knowledge_time has arrived, so
    # feature computation and (crucially) prediction grading stay strictly point-in-time.
    revealed = sorted(observations, key=lambda o: o.knowledge_time)
    store = InMemoryBitemporalStore()
    feature_store = FeatureStore(store, list(features))
    cursor = 0

    pieces: list[pd.Series] = []
    ic_rows: dict[datetime, dict[str, float]] = {}
    prev_w: dict[str, float] = {}
    for i, t in enumerate(rebalances):
        chunk = []
        while cursor < len(revealed) and revealed[cursor].knowledge_time <= t:
            chunk.append(revealed[cursor])
            cursor += 1
        if chunk:
            store.add_many(chunk)

        matrix = feature_store.compute_matrix(t, sorted(names))
        graded = grade_predictions(
            store, t, horizon_days=horizon_days, model_version=model_version, min_abs_ic=min_abs_ic
        )
        ic = {str(k): float(v) for k, v in graded.get("ic", {}).items()}
        if ic:
            ic_rows[t] = ic

        scorer = ic_weighted_scorer(ic, min_abs_ic=min_abs_ic) if ic else composite_scorer()
        scores = scorer(matrix, t) if not matrix.empty else pd.Series(dtype=float)
        ranked = scores.dropna()
        if not ranked.empty:
            log_cycle_predictions(
                store, ranked, matrix, t, model_version=model_version, horizon_days=horizon_days
            )

        trailing = full_returns.loc[full_returns.index <= t]
        weights = size_book(
            trailing,
            target_vol=target_vol,
            limits=limits,
            scores=ranked if tilt_strength > 0 else None,
            tilt_strength=tilt_strength,
        )
        cost = cost_model.turnover_cost(prev_w, weights)

        t_next = rebalances[i + 1] if i + 1 < len(rebalances) else full_returns.index[-1]
        window = full_returns.loc[(full_returns.index > t) & (full_returns.index <= t_next)]
        if not window.empty and weights:
            w_series = pd.Series(weights, dtype=float).reindex(window.columns).fillna(0.0)
            contrib = window.fillna(0.0).mul(w_series, axis=1).sum(axis=1)
            if cost and len(contrib) > 0:
                contrib.iloc[0] -= cost  # charge the turnover cost on the first day of the window
            pieces.append(contrib)
        prev_w = weights

    net = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)
    if net.empty:
        raise ValueError("replay produced no returns (check schedule/horizon vs history length)")
    equity = (1.0 + net).cumprod()
    ic_history = pd.DataFrame(ic_rows).T.sort_index()
    summary = summarize(net, turnover=pd.Series(dtype=float), periods_per_year=TRADING_DAYS)
    return ReplayResult(
        equity_curve=equity, net_returns=net, ic_history=ic_history, summary=summary
    )
