"""Phase 0 end-to-end demo.

Builds a seeded synthetic universe in the bitemporal store, shows the
knowledge-time visibility growing over time, then runs the harness: a toy
momentum strategy vs. the equal-weight and buy-and-hold baselines, **net of
costs**. The numbers are meaningless as alpha (toy data) — the point is that the
whole pipeline runs honestly end to end.
"""

from __future__ import annotations

from datetime import date

from market_trader.backtest import (
    BasicCostModel,
    MomentumStrategy,
    summaries_to_frame,
)
from market_trader.backtest.engine import compare_to_baselines
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.observability import configure_logging, get_logger
from market_trader.storage import InMemoryBitemporalStore

COLUMNS = [
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "hit_rate",
    "avg_turnover",
]


def main() -> None:
    configure_logging("INFO")
    log = get_logger("phase0_demo")

    symbols = [f"SYM{i:02d}" for i in range(12)]
    observations = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=180, seed=7
    )
    store = InMemoryBitemporalStore()
    store.add_many(observations)
    log.info("loaded_observations", count=store.count(), symbols=len(symbols))

    event_days = sorted({o.event_time for o in observations})
    early, late = event_days[20], event_days[-1]
    log.info(
        "knowledge_time_visibility",
        early_date=early.date().isoformat(),
        rows_visible_early=len(store.as_of(early)),
        late_date=late.date().isoformat(),
        rows_visible_late=len(store.as_of(late)),
    )

    schedule = event_days[25:-1:5]  # weekly rebalance after a warmup, leaving forward room
    summaries = compare_to_baselines(
        store, MomentumStrategy(lookback=20), schedule, BasicCostModel()
    )
    frame = summaries_to_frame(summaries)[COLUMNS]

    print("\nPhase 0 harness — net of costs, vs. baselines:\n")
    print(frame.round(3).to_string())
    print("\n(Toy synthetic data: a plumbing demo, not a claim of edge.)")


if __name__ == "__main__":
    main()
