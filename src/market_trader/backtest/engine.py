"""The backtest engine.

For each rebalance date the strategy sees only a point-in-time view (knowledge
time ≤ that date) and returns target weights. The engine accounts realised P&L
using *forward* returns the strategy never saw, charges costs on turnover, and
summarises net performance. Baselines are computed the same way so comparisons
are apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_trader.backtest.costs import BasicCostModel, CostModel, one_way_turnover
from market_trader.backtest.metrics import TRADING_DAYS, PerformanceSummary, summarize
from market_trader.backtest.pit import StorePriceView, observations_to_price_frame
from market_trader.backtest.strategies import EqualWeightStrategy
from market_trader.backtest.types import Strategy, Weights
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE
from market_trader.storage.bitemporal import BitemporalStore


@dataclass
class BacktestResult:
    strategy_name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    equity_curve: pd.Series
    turnover: pd.Series
    summary: PerformanceSummary


def _realized_returns(store: BitemporalStore, dataset: str, price_field: str) -> pd.DataFrame:
    """Daily returns over the *full* history, for P&L accounting only.

    This is the "what actually happened" panel — never handed to a strategy.
    """
    obs = store.as_of(DISTANT_FUTURE, dataset=dataset)
    panel = observations_to_price_frame(obs, price_field)
    if panel.empty:
        return panel
    return panel.pct_change()


def run_backtest(
    store: BitemporalStore,
    strategy: Strategy,
    schedule: Sequence[datetime],
    cost_model: CostModel | None = None,
    *,
    dataset: str = PRICE_DATASET,
    price_field: str = "close",
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    if not schedule:
        raise ValueError("schedule must be non-empty")
    cost_model = cost_model or BasicCostModel()
    rebalances = sorted(schedule)

    returns = _realized_returns(store, dataset, price_field)
    if returns.empty:
        raise ValueError("no price data in store")
    index = returns.index

    gross = pd.Series(0.0, index=index)
    costs = pd.Series(0.0, index=index)
    turnover_by_date: dict[datetime, float] = {}

    prev_w: Weights = {}
    for i, t in enumerate(rebalances):
        view = StorePriceView(store, as_of_time=t, dataset=dataset, price_field=price_field)
        w = strategy.target_weights(view, t)
        turnover_by_date[t] = one_way_turnover(prev_w, w)
        cost = cost_model.turnover_cost(prev_w, w)

        t_next = rebalances[i + 1] if i + 1 < len(rebalances) else index[-1]
        window = returns.loc[(returns.index > t) & (returns.index <= t_next)]
        if not window.empty:
            w_series = pd.Series(w, dtype=float).reindex(returns.columns).fillna(0.0)
            contrib = window.fillna(0.0).mul(w_series, axis=1).sum(axis=1)
            gross.loc[contrib.index] = contrib.to_numpy()
            costs.loc[window.index[0]] += cost
        prev_w = w

    net = gross - costs
    active = index > rebalances[0]
    net_active = net[active]
    gross_active = gross[active]
    equity = (1.0 + net_active).cumprod()
    turnover = pd.Series(turnover_by_date).sort_index()
    summary = summarize(net_active, turnover=turnover, periods_per_year=periods_per_year)

    return BacktestResult(
        strategy_name=strategy.name,
        net_returns=net_active,
        gross_returns=gross_active,
        equity_curve=equity,
        turnover=turnover,
        summary=summary,
    )


def buy_and_hold_summary(
    store: BitemporalStore,
    *,
    dataset: str = PRICE_DATASET,
    price_field: str = "close",
    periods_per_year: int = TRADING_DAYS,
    start_after: datetime | None = None,
) -> PerformanceSummary:
    """Equal-initial-weight buy-and-hold of the whole universe (no rebalancing)."""
    returns = _realized_returns(store, dataset, price_field)
    if returns.empty:
        raise ValueError("no price data in store")
    if start_after is not None:
        returns = returns.loc[returns.index > start_after]
    equity_per_symbol = (1.0 + returns.fillna(0.0)).cumprod()
    portfolio = equity_per_symbol.mean(axis=1)
    daily = portfolio.pct_change().fillna(0.0)
    return summarize(daily, turnover=pd.Series(dtype=float), periods_per_year=periods_per_year)


def compare_to_baselines(
    store: BitemporalStore,
    strategy: Strategy,
    schedule: Sequence[datetime],
    cost_model: CostModel | None = None,
    *,
    dataset: str = PRICE_DATASET,
    price_field: str = "close",
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, PerformanceSummary]:
    """Run ``strategy`` and the equal-weight / buy-and-hold baselines, net of costs."""
    rebalances = sorted(schedule)
    candidate = run_backtest(
        store,
        strategy,
        rebalances,
        cost_model,
        dataset=dataset,
        price_field=price_field,
        periods_per_year=periods_per_year,
    )
    equal_weight = run_backtest(
        store,
        EqualWeightStrategy(),
        rebalances,
        cost_model,
        dataset=dataset,
        price_field=price_field,
        periods_per_year=periods_per_year,
    )
    bnh = buy_and_hold_summary(
        store,
        dataset=dataset,
        price_field=price_field,
        periods_per_year=periods_per_year,
        start_after=rebalances[0],
    )

    out: dict[str, PerformanceSummary] = {candidate.strategy_name: candidate.summary}
    out.setdefault("equal_weight", equal_weight.summary)
    out["buy_and_hold"] = bnh
    return out


def summaries_to_frame(summaries: dict[str, PerformanceSummary]) -> pd.DataFrame:
    return pd.DataFrame({name: s.as_dict() for name, s in summaries.items()}).T
