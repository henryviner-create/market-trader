"""Forecasting tier: models, calibration, stacking, PIT dataset, and pipeline."""

from __future__ import annotations

from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score

from market_trader.backtest.engine import compare_to_baselines, run_backtest
from market_trader.backtest.metrics import brier_score
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.features import FeatureStore, default_features
from market_trader.forecasting import (
    REGIME_FEATURE,
    Forecaster,
    ForecastStrategy,
    build_training_set,
    evaluate_predictions,
    make_calibrated,
    make_gbt,
    make_logistic,
    make_stacking,
    purged_cv_auc,
    time_split,
    train_forecaster,
)
from market_trader.storage import InMemoryBitemporalStore
from market_trader.universe import PointInTimeUniverse


def test_base_models_learn_a_separable_signal() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(int)
    for make in (make_logistic, make_gbt):
        f = Forecaster(make(), "m").fit(X[:300], y[:300])
        p = f.predict_proba(X[300:])
        assert p.min() >= 0.0 and p.max() <= 1.0  # probabilities, never point preds
        assert roc_auc_score(y[300:], p) > 0.8


def test_calibration_does_not_worsen_brier() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (2000, 2))
    y = (X[:, 0] + rng.normal(0, 1, 2000) > 0).astype(int)
    raw = Forecaster(make_gbt(), "raw").fit(X[:1500], y[:1500])
    cal = Forecaster(make_calibrated(make_gbt(), cv=3), "cal").fit(X[:1500], y[:1500])
    p_raw, p_cal = raw.predict_proba(X[1500:]), cal.predict_proba(X[1500:])
    assert p_cal.min() >= 0.0 and p_cal.max() <= 1.0
    assert brier_score(p_cal, y[1500:]) <= brier_score(p_raw, y[1500:]) + 0.02


def test_stacking_learns() -> None:
    rng = np.random.default_rng(2)
    f1, f2 = rng.normal(0, 1, 1200), rng.normal(0, 1, 1200)
    X = np.column_stack([f1, f2, rng.normal(0, 1, 1200)])
    y = (f1 + f2 + rng.normal(0, 0.5, 1200) > 0).astype(int)
    stack = Forecaster(make_stacking(), "s").fit(X[:900], y[:900])
    assert roc_auc_score(y[900:], stack.predict_proba(X[900:])) > 0.75


def _synthetic_store(
    n_syms: int = 8, n_days: int = 220
) -> tuple[InMemoryBitemporalStore, list, PointInTimeUniverse]:
    syms = [f"S{i}" for i in range(n_syms)]
    obs = synthetic_price_observations(symbols=syms, start=date(2022, 1, 3), n_days=n_days, seed=5)
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    universe = PointInTimeUniverse.from_records(
        [{"symbol": s, "name": s, "added": "2000-01-01", "removed": None} for s in syms]
    )
    days = sorted({o.event_time for o in obs})
    return store, days, universe


def test_training_set_is_point_in_time_and_labelled() -> None:
    store, days, universe = _synthetic_store()
    fs = FeatureStore(store, default_features())
    schedule = days[65:-10:5]
    ts = build_training_set(store, fs, schedule, universe=universe, horizon_days=5)

    assert len(ts) > 0
    assert REGIME_FEATURE in ts.feature_names
    assert set(ts.y.unique()) <= {0, 1}
    assert (ts.t1 > ts.t0).all()  # label realised strictly after the decision


def test_pipeline_trains_evaluates_and_backtests_vs_baselines() -> None:
    store, days, universe = _synthetic_store()
    fs = FeatureStore(store, default_features())
    schedule = days[65:-10:5]
    train_dates, test_dates = time_split(schedule, 0.6)

    model, ts = train_forecaster(
        store, fs, train_dates, universe=universe, make_model=make_logistic, horizon_days=5
    )
    strategy = ForecastStrategy(store, fs, model, universe, feature_names=ts.feature_names)

    result = run_backtest(store, strategy, test_dates)
    assert result.summary.n_periods > 0
    assert result.equity_curve.notna().all()

    summaries = compare_to_baselines(store, strategy, test_dates)
    assert {"forecast", "equal_weight", "buy_and_hold"} <= set(summaries)

    # Honest evaluation: probabilities + calibration report + purged-CV AUC (no edge claim).
    probs = model.predict_proba(ts.X.to_numpy(dtype=float))
    report = evaluate_predictions(probs, ts.y.to_numpy())
    assert len(report.reliability) == 10
    auc = purged_cv_auc(ts, make_logistic, n_splits=4)
    assert np.isnan(auc) or 0.0 <= auc <= 1.0
