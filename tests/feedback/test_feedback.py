"""Feedback loop: prediction logging, scoring, drift, pruning."""

from __future__ import annotations

from datetime import date

from market_trader.core.time import day_close
from market_trader.feedback import (
    DriftMonitor,
    PredictionRecord,
    load_predictions,
    log_predictions,
    prune_signals_by_ic,
    score_predictions,
)
from market_trader.storage import InMemoryBitemporalStore


def test_prediction_log_round_trips_snapshot_and_respects_knowledge_time() -> None:
    store = InMemoryBitemporalStore()
    rec = PredictionRecord(
        as_of=day_close(date(2023, 1, 10)),
        symbol="AAPL",
        probability=0.7,
        horizon_days=5,
        model_version="v1",
        features={"mom_60": 1.2, "vol_20": 0.3},
    )
    log_predictions(store, [rec])

    loaded = load_predictions(store, day_close(date(2023, 1, 15)))
    assert len(loaded) == 1
    assert loaded[0].features == {"mom_60": 1.2, "vol_20": 0.3}  # full snapshot reproduced
    assert abs(loaded[0].probability - 0.7) < 1e-9

    log_predictions(store, [rec])  # idempotent
    assert len(load_predictions(store, day_close(date(2023, 1, 15)))) == 1
    assert load_predictions(store, day_close(date(2023, 1, 5))) == []  # not knowable earlier


def test_score_predictions_known_values() -> None:
    t = day_close(date(2023, 1, 10))
    preds = [
        PredictionRecord(t, "A", 0.8, 5, "v"),
        PredictionRecord(t, "B", 0.3, 5, "v"),
    ]
    scored = score_predictions(preds, {"A": 0.05, "B": -0.02})
    assert scored["n"] == 2
    assert scored["hit_rate"] == 1.0  # 0.8->up correct, 0.3->down correct
    assert abs(scored["brier"] - ((0.2**2 + 0.3**2) / 2)) < 1e-12


def test_drift_monitor_fires_outside_tolerance() -> None:
    monitor = DriftMonitor(baseline=0.55, tolerance=0.10)
    assert monitor.is_drifting(0.40) is True
    assert monitor.is_drifting(0.53) is False


def test_prune_signals_by_ic() -> None:
    kept, pruned = prune_signals_by_ic({"mom": 0.05, "dead": 0.001, "neg": -0.08}, min_abs_ic=0.02)
    assert kept == ["mom", "neg"]
    assert pruned == ["dead"]
