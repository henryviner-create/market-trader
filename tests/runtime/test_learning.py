"""The learning loop: log predictions, then grade them against realised outcomes."""

from __future__ import annotations

from datetime import date

from market_trader.core.synthetic import synthetic_price_observations
from market_trader.features import FeatureStore, default_features
from market_trader.feedback.prediction_log import load_predictions
from market_trader.portfolio import composite_score, equal_weights
from market_trader.runtime.learning import grade_predictions, log_cycle_predictions
from market_trader.storage import InMemoryBitemporalStore


def _store(symbols: list[str], n_days: int = 120):
    obs = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=5
    )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    return store, sorted({o.event_time for o in obs})


def _scores(store, as_of, symbols):
    fs = FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)
    return composite_score(matrix, equal_weights(matrix.columns)), matrix


def test_log_and_grade_predictions_roundtrip() -> None:
    symbols = [f"S{i}" for i in range(10)]
    store, dates = _store(symbols)
    as_of_early, end = dates[-15], dates[-1]  # leave room for the 5-day forward window
    scores, matrix = _scores(store, as_of_early, symbols)

    assert log_cycle_predictions(store, scores, matrix, as_of_early, model_version="test") > 0
    assert load_predictions(store, end, model_version="test")  # persisted + reloadable

    res = grade_predictions(store, end, model_version="test")
    assert res["n"] > 0  # outcomes known -> graded
    assert 0.0 <= res["brier"] <= 1.0
    assert isinstance(res["ic"], dict) and res["ic"]  # per-signal IC computed


def test_grade_requires_the_full_horizon() -> None:
    symbols = [f"S{i}" for i in range(6)]
    store, dates = _store(symbols)
    partial = dates[-3]  # only 2 bars ahead; the 5-day horizon hasn't elapsed
    scores, matrix = _scores(store, partial, symbols)
    log_cycle_predictions(store, scores, matrix, partial, model_version="test")
    # Must not grade against a half-formed window -> no fake numbers on fresh logs.
    assert grade_predictions(store, dates[-1], model_version="test", horizon_days=5)["n"] == 0
