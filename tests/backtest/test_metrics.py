"""Unit tests for performance and calibration metrics (known-value checks)."""

from __future__ import annotations

import numpy as np

from market_trader.backtest.metrics import (
    annualized_return,
    bootstrap_ci,
    brier_score,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    reliability_curve,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)


def test_max_drawdown_known_value() -> None:
    # equity: 1.10 then 0.55 -> worst drawdown = 0.55/1.10 - 1 = -0.5
    assert abs(max_drawdown([0.10, -0.50, 0.0]) - (-0.5)) < 1e-12


def test_sharpe_matches_formula_with_sample_std() -> None:
    r = np.array([0.02, 0.01, 0.03, 0.02])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(sharpe_ratio(r) - expected) < 1e-9


def test_sharpe_zero_variance_is_zero_not_inf() -> None:
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sortino_only_penalises_downside() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02])
    excess = r
    downside = excess[excess < 0]
    expected = excess.mean() / np.sqrt(np.mean(downside**2)) * np.sqrt(252)
    assert abs(sortino_ratio(r) - expected) < 1e-9


def test_calmar_is_return_over_drawdown() -> None:
    r = [0.05, -0.10, 0.04, 0.03]
    expected = annualized_return(r) / abs(max_drawdown(r))
    assert abs(calmar_ratio(r) - expected) < 1e-9


def test_annualized_return_quarterly() -> None:
    r = [0.05, 0.05, 0.05, 0.05]
    assert abs(annualized_return(r, periods_per_year=4) - (1.05**4 - 1)) < 1e-9


def test_hit_rate_counts_strictly_positive() -> None:
    assert hit_rate([1.0, -1.0, 2.0, -3.0, 0.0]) == 0.4


def test_brier_score() -> None:
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert abs(brier_score([0.5, 0.5], [1.0, 0.0]) - 0.25) < 1e-12


def test_reliability_curve_bins_and_counts() -> None:
    probs = [0.05, 0.15, 0.95, 0.92]
    outcomes = [0.0, 0.0, 1.0, 1.0]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 10
    assert int(curve["count"].sum()) == 4
    top = curve.iloc[-1]
    assert top["count"] == 2
    assert abs(top["frac_pos"] - 1.0) < 1e-12


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.015])
    lo1, hi1 = bootstrap_ci(r, seed=42, n_boot=200)
    lo2, hi2 = bootstrap_ci(r, seed=42, n_boot=200)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_summarize_shapes() -> None:
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    s = summarize(r, turnover=[0.0, 0.5, 0.5], periods_per_year=252)
    assert s.n_periods == 5
    assert abs(s.avg_turnover - (1.0 / 3.0)) < 1e-9
    assert set(s.as_dict()) == {
        "n_periods",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "avg_turnover",
    }
