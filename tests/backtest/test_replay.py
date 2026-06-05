"""Walk-forward replay of the learning system — it runs PIT and advances the IC loop."""

from __future__ import annotations

from datetime import date

from market_trader.backtest.replay import replay_learning
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.features.technical import MeanReversion, Momentum, Volatility


def test_replay_learning_runs_pit_and_tracks_measured_ic() -> None:
    syms = [f"S{i}" for i in range(8)]
    obs = synthetic_price_observations(symbols=syms, start=date(2022, 1, 3), n_days=400, seed=4)
    times = sorted({o.knowledge_time for o in obs if o.entity_id == "S0"})
    schedule = times[60::21]  # ~16 monthly rebalances
    features = [Momentum(lookback=20), MeanReversion(lookback=5), Volatility(window=20)]

    result = replay_learning(
        obs,
        universe=syms,
        schedule=schedule,
        features=features,
        horizon_days=21,
        tilt_strength=1.0,
    )

    assert len(result.net_returns) > 0
    assert result.equity_curve.iloc[-1] > 0  # a finite, positive equity path
    assert "sharpe" in result.summary.as_dict()  # summary computed
    # The learning loop graded its own matured predictions and measured per-signal IC over
    # time — the live-IC track record (the overfit detector). Columns are the signal names.
    assert not result.ic_history.empty
    assert any(c.startswith(("mom", "meanrev", "vol")) for c in result.ic_history.columns)


def test_replay_learning_needs_enough_history() -> None:
    syms = ["A", "B"]
    obs = synthetic_price_observations(symbols=syms, start=date(2022, 1, 3), n_days=30, seed=1)
    times = sorted({o.knowledge_time for o in obs if o.entity_id == "A"})
    try:
        replay_learning(obs, universe=syms, schedule=times[:1], features=[Momentum(lookback=5)])
        raise AssertionError("expected a ValueError for too few rebalances")
    except ValueError:
        pass  # needs >= 2 rebalance dates
