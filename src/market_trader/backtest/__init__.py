"""Validation / backtest harness.

Built in-house (not zipline/backtrader/vectorbt) for one reason: the harness
replays the world strictly through *knowledge time* from the bitemporal store, so
lookahead bias is structurally hard rather than a thing you have to remember not
to do. See ``DECISIONS.md`` D4.
"""

from market_trader.backtest.costs import BasicCostModel, CostModel, ZeroCostModel, one_way_turnover
from market_trader.backtest.engine import (
    BacktestResult,
    buy_and_hold_summary,
    compare_to_baselines,
    run_backtest,
    summaries_to_frame,
)
from market_trader.backtest.metrics import PerformanceSummary, summarize
from market_trader.backtest.pit import StorePriceView, observations_to_price_frame
from market_trader.backtest.splitters import PurgedKFold, walk_forward
from market_trader.backtest.strategies import EqualWeightStrategy, MomentumStrategy
from market_trader.backtest.types import PointInTimeView, Strategy, Weights

__all__ = [
    "BacktestResult",
    "BasicCostModel",
    "CostModel",
    "EqualWeightStrategy",
    "MomentumStrategy",
    "PerformanceSummary",
    "PointInTimeView",
    "PurgedKFold",
    "StorePriceView",
    "Strategy",
    "Weights",
    "ZeroCostModel",
    "buy_and_hold_summary",
    "compare_to_baselines",
    "observations_to_price_frame",
    "one_way_turnover",
    "run_backtest",
    "summaries_to_frame",
    "summarize",
    "walk_forward",
]
