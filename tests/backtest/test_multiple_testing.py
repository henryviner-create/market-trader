"""Multiple-testing correction: don't promote a lucky subset of a signal zoo."""

from __future__ import annotations

import pytest

from market_trader.backtest.multiple_testing import (
    benjamini_hochberg,
    bonferroni_t_threshold,
    correct_family,
    normal_two_sided_p,
)


def test_bonferroni_bar_rises_with_more_trials() -> None:
    assert bonferroni_t_threshold(1) == pytest.approx(1.96, abs=0.01)
    assert bonferroni_t_threshold(20) > bonferroni_t_threshold(1)  # more looks -> higher bar
    assert bonferroni_t_threshold(20) == pytest.approx(3.02, abs=0.03)


def test_two_sided_p() -> None:
    assert normal_two_sided_p(0.0) == pytest.approx(1.0)
    assert normal_two_sided_p(1.96) == pytest.approx(0.05, abs=0.005)


def test_benjamini_hochberg_picks_the_real_one() -> None:
    sig = benjamini_hochberg({"strong": 0.0001, "a": 0.4, "b": 0.6, "c": 0.8}, alpha=0.05)
    assert sig["strong"]
    assert not (sig["a"] or sig["b"] or sig["c"])


def test_correct_family_rejects_a_raw_significant_fluke() -> None:
    # A zoo of 10: one genuinely strong, one that clears a bare |t|>=2, the rest noise.
    t_stats = {"strong": 4.5, "lucky": 2.1, **{f"n{i}": 0.3 for i in range(8)}}
    fam = correct_family(t_stats, alpha=0.05)

    assert fam.n_trials == 10
    assert fam.bonferroni["strong"]  # survives the strict family-wise bar
    assert not fam.bonferroni["lucky"]  # raw-significant, but fails once breadth is accounted for
    assert fam.fdr["strong"] and not fam.fdr["lucky"]  # FDR agrees the fluke is not a discovery
