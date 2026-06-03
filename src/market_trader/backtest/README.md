# backtest — the validation harness

Built in-house so point-in-time replay is structural, not a thing you remember to
do. A strategy is handed **only** a view of the world as it was knowable at the
decision time; the engine accounts P&L with forward returns the strategy never
saw.

## Pieces

- **`pit.py`** — `StorePriceView`: a point-in-time view over the bitemporal store
  at a fixed `as_of`. It cannot surface a price that wasn't yet knowable.
- **`types.py`** — `Strategy` (maps a view → target `Weights`) and the
  `PointInTimeView` protocol.
- **`strategies.py`** — `EqualWeightStrategy` (a baseline) and `MomentumStrategy`
  (a toy candidate — not a claim of edge).
- **`costs.py`** — `BasicCostModel` charges commission + half-spread + slippage on
  one-way turnover. A backtest without costs reports an upper bound.
- **`splitters.py`** — `walk_forward` and `PurgedKFold` (purges training samples
  whose label horizon overlaps the test fold, plus an embargo) to stop leakage
  across CV boundaries.
- **`metrics.py`** — Sharpe/Sortino/Calmar/max-drawdown/hit-rate/turnover, plus
  **calibration** (reliability curve + Brier) and **bootstrap CIs**. Degenerate
  inputs degrade to `0.0`, never `inf`/`nan`.
- **`engine.py`** — `run_backtest`, `buy_and_hold_summary`, and
  `compare_to_baselines` (candidate vs equal-weight vs buy-and-hold, net of
  costs).

## Usage

```python
from market_trader.backtest import BasicCostModel, MomentumStrategy
from market_trader.backtest.engine import compare_to_baselines, summaries_to_frame

summaries = compare_to_baselines(store, MomentumStrategy(), schedule, BasicCostModel())
print(summaries_to_frame(summaries).round(3))
```

## Adding a strategy

Implement the `Strategy` protocol:

```python
@dataclass
class MyStrategy:
    name: str = "my_strategy"

    def target_weights(self, view, as_of):
        prices = view.price_panel()      # only knowledge_time <= as_of
        ...                              # return {symbol: weight}
```

The engine and metrics are agnostic to how weights are formed — which is where
the signal, forecasting, and weighting tiers will plug in.

## Honesty checks

`tests/leakage/test_no_lookahead.py` proves that inserting future data does not
change any past decision or overlapping P&L — the harness-level complement to the
storage guarantee.
