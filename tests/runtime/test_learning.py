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


def test_grade_is_empty_until_the_horizon_elapses() -> None:
    symbols = [f"S{i}" for i in range(6)]
    store, dates = _store(symbols)
    latest = dates[-1]
    scores, matrix = _scores(store, latest, symbols)
    # Logged at the very last bar -> no forward returns exist yet to grade against.
    log_cycle_predictions(store, scores, matrix, latest, model_version="test")
    assert grade_predictions(store, latest, model_version="test", horizon_days=5)["n"] == 0
