"""Pluggable scorers: the forecaster trains + scores, build_scorer dispatches, and
the out-of-sample gate reports both models — all offline on a synthetic store.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from market_trader.config import Settings
from market_trader.core.synthetic import synthetic_price_observations
from market_trader.features import FeatureStore, default_features
from market_trader.forecasting.models import make_logistic
from market_trader.runtime.scoring import (
    build_scorer,
    composite_scorer,
    forecast_scorer,
    forecaster_vs_baseline_auc,
    ic_weighted_scorer,
)
from market_trader.storage import InMemoryBitemporalStore

_AT = datetime(2024, 1, 1)


def _store(symbols: list[str], n_days: int = 170):
    obs = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=11
    )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    return store, max(o.event_time for o in obs)


def test_forecast_scorer_trains_and_returns_probabilities() -> None:
    symbols = [f"S{i}" for i in range(12)]
    store, as_of = _store(symbols)
    fs = FeatureStore(store, default_features())

    scorer = forecast_scorer(store, fs, symbols, as_of, make_model=make_logistic)
    scores = scorer(fs.compute_matrix(as_of, symbols), as_of)

    assert list(scores.index) == symbols
    assert scores.between(0.0, 1.0).all()  # P(out-perform), bounded


def test_build_scorer_defaults_to_composite() -> None:
    symbols = [f"S{i}" for i in range(8)]
    store, as_of = _store(symbols)
    fs = FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)

    chosen = build_scorer(Settings(scorer="composite"), store, fs, symbols, as_of)
    assert chosen(matrix, as_of).equals(composite_scorer()(matrix, as_of))


def test_ic_weighting_falls_back_to_equal_weight_on_cold_start() -> None:
    # No graded IC yet -> identical to the equal-weight composite (never worse).
    matrix = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}, index=["X", "Y", "Z"])
    cold = ic_weighted_scorer({}, min_abs_ic=0.02)(matrix, _AT)
    pd.testing.assert_series_equal(cold, composite_scorer()(matrix, _AT))


def test_ic_weighting_prunes_weak_signals_and_inverts_negative_ic() -> None:
    # 'good' (+IC) ranks A>C; 'bad' has the opposite raw pattern but -IC, so once
    # inverted it agrees with 'good'; 'dead' is below the floor and ignored.
    matrix = pd.DataFrame(
        {"good": [3.0, 2.0, 1.0], "bad": [1.0, 2.0, 3.0], "dead": [5.0, 1.0, 9.0]},
        index=["A", "B", "C"],
    )
    scores = ic_weighted_scorer({"good": 0.30, "bad": -0.30, "dead": 0.005}, min_abs_ic=0.02)(
        matrix, _AT
    )
    assert scores["A"] > scores["B"] > scores["C"]


def test_build_scorer_uses_ic_when_supplied_and_respects_the_toggle() -> None:
    symbols = [f"S{i}" for i in range(8)]
    store, as_of = _store(symbols)
    fs = FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)
    ic = {str(matrix.columns[0]): 0.5}  # only the first signal carries IC

    used = build_scorer(Settings(scorer="composite"), store, fs, symbols, as_of, ic=ic)
    assert not used(matrix, as_of).equals(composite_scorer()(matrix, as_of))  # IC reshaped it
    pd.testing.assert_series_equal(
        used(matrix, as_of), ic_weighted_scorer(ic, min_abs_ic=Settings().ic_min_abs)(matrix, as_of)
    )

    off = build_scorer(
        Settings(scorer="composite", ic_weighting=False), store, fs, symbols, as_of, ic=ic
    )
    pd.testing.assert_series_equal(off(matrix, as_of), composite_scorer()(matrix, as_of))


def test_forecaster_vs_baseline_auc_reports_both_models() -> None:
    symbols = [f"S{i}" for i in range(14)]
    store, as_of = _store(symbols)

    res = forecaster_vs_baseline_auc(store, symbols, as_of, forecast_model=make_logistic)

    assert set(res) == {"n_samples", "forecast_cv_auc", "baseline_auc"}
    assert res["n_samples"] > 0  # built point-in-time samples to score against
