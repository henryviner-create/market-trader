"""Weighting and portfolio-construction math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_trader.portfolio import (
    composite_score,
    fractional_kelly_weights,
    hierarchical_risk_parity,
    ic_weights,
    information_coefficient,
    inverse_variance_weights,
    ledoit_wolf_cov,
    min_variance_weights,
    orthogonality_penalty,
    risk_parity_weights,
    volatility_target_weights,
)


def test_information_coefficient_perfect_and_reversed() -> None:
    sig = pd.Series([1, 2, 3, 4, 5], index=list("abcde"), dtype=float)
    fwd = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=list("abcde"))
    assert abs(information_coefficient(sig, fwd) - 1.0) < 1e-9
    assert abs(information_coefficient(sig, -fwd) + 1.0) < 1e-9


def test_composite_score_orders_by_weighted_signal() -> None:
    matrix = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0]}, index=["a", "b", "c"])
    score = composite_score(matrix, pd.Series({"f1": 1.0, "f2": 0.0}))
    assert score["a"] < score["b"] < score["c"]


def test_ic_weights_are_sign_aware_and_capped() -> None:
    w = ic_weights(pd.Series({"a": 0.4, "b": -0.4}), cap=0.5)
    assert w["a"] > 0 and w["b"] < 0
    assert abs(w).max() <= 0.5 + 1e-9


def test_inverse_variance_favours_low_variance() -> None:
    m = pd.DataFrame({"low": [1.0, 1.0, 1.0, 1.001], "high": [0.0, 5.0, -5.0, 2.0]})
    w = inverse_variance_weights(m)
    assert w["low"] > w["high"]
    assert abs(w.sum() - 1.0) < 1e-9


def test_orthogonality_penalises_redundant_signals() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    m = pd.DataFrame({"a": x, "b": x + rng.normal(0, 1e-6, 200), "c": rng.normal(0, 1, 200)})
    penalty = orthogonality_penalty(m)
    assert penalty["c"] > penalty["a"]  # independent signal keeps more weight


def test_volatility_target_scales_to_target() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5})
    cov = pd.DataFrame(np.diag([0.0004, 0.0004]), index=["A", "B"], columns=["A", "B"])
    scaled = volatility_target_weights(w, cov, target_vol=0.15, periods_per_year=252).to_numpy()
    realised = float(np.sqrt(scaled @ cov.to_numpy() @ scaled) * np.sqrt(252))
    assert abs(realised - 0.15) < 1e-9


def test_fractional_kelly_tilts_to_higher_edge() -> None:
    cov = pd.DataFrame(np.eye(2) * 0.01, index=["A", "B"], columns=["A", "B"])
    w = fractional_kelly_weights(pd.Series({"A": 0.02, "B": 0.01}), cov, fraction=0.25)
    assert w["A"] > w["B"] > 0


def test_risk_parity_and_min_variance_favour_low_vol() -> None:
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["H", "L"], columns=["H", "L"])
    rp = risk_parity_weights(cov)
    assert rp["L"] > rp["H"] and abs(rp.sum() - 1.0) < 1e-6
    mv = min_variance_weights(cov)
    assert mv["L"] > mv["H"] and abs(mv.sum() - 1.0) < 1e-9


def test_ledoit_wolf_is_symmetric_psd() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 4)), columns=list("abcd"))
    cov = ledoit_wolf_cov(returns).to_numpy()
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_hrp_produces_valid_long_only_weights() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 5)), columns=list("abcde"))
    w = hierarchical_risk_parity(returns)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w > 0).all()
    assert list(w.index) == list(returns.columns)
