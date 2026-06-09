"""Multiple-testing correction for the signal gate.

Testing a *zoo* of candidate signals and promoting any that clears a bare ``|t| >= 2`` is how
you manufacture fake edges: at alpha=0.05 you expect ~1 false "significant" hit per 20 signals
by chance alone. This applies the two standard family-wise corrections so a signal must clear a
bar that *accounts for how many signals were tried*:

* **Bonferroni** controls the family-wise error rate (P(any false positive)) — the strict bar:
  a signal needs ``|t| >= Phi^-1(1 - alpha/2m)`` for ``m`` trials.
* **Benjamini-Hochberg** controls the false-discovery rate (expected fraction of promoted
  signals that are false) — less strict, the usual choice for a research screen.

Large-sample normal approximation for the per-date-averaged IC t-stat (``statistics.NormalDist``
— stdlib, no new dependency). The honest holdout (``backtest/holdout.py``) is the complementary
guard: this stops you being fooled by *breadth* of search, the holdout by *repetition* of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


def normal_two_sided_p(t_stat: float) -> float:
    """Two-sided p-value for a (large-sample) t-like statistic under the normal approximation."""
    return float(2.0 * (1.0 - _N.cdf(abs(t_stat))))


def bonferroni_t_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """The two-sided ``|t|`` a single signal must exceed for family-wise significance at
    ``alpha`` across ``n_trials`` looks (Bonferroni; normal approximation)."""
    m = max(1, n_trials)
    return float(_N.inv_cdf(1.0 - alpha / (2.0 * m)))


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Benjamini-Hochberg FDR: which hypotheses are significant controlling the expected
    false-discovery rate at ``alpha``. Returns ``{name: significant}``."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])  # ascending p
    m = len(items)
    if m == 0:
        return {}
    k = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            k = i  # largest rank meeting the BH line -> reject ranks 1..k
    significant = {name for rank, (name, _) in enumerate(items, start=1) if rank <= k}
    return {name: (name in significant) for name in pvalues}


@dataclass(frozen=True)
class FamilyResult:
    """Per-signal significance verdicts after correcting for the whole family tested."""

    n_trials: int
    alpha: float
    bonferroni_t: float  # the |t| bar a signal must clear (Bonferroni)
    pvalues: dict[str, float]
    bonferroni: dict[str, bool]  # family-wise significant (strict)
    fdr: dict[str, bool]  # FDR-significant (Benjamini-Hochberg)


def correct_family(t_stats: dict[str, float], *, alpha: float = 0.05) -> FamilyResult:
    """Apply Bonferroni + BH-FDR across all signals tested together.

    ``t_stats`` maps signal -> its IC t-stat; ``n_trials`` is len(t_stats). A signal that clears
    a bare ``|t| >= 2`` can still fail both corrections once the breadth of search is accounted
    for — which is exactly the discipline that stops a lucky subset being promoted.
    """
    m = len(t_stats)
    bonf_t = bonferroni_t_threshold(m, alpha)
    pvals = {name: normal_two_sided_p(t) for name, t in t_stats.items()}
    bonf = {name: abs(t) >= bonf_t for name, t in t_stats.items()}
    fdr = benjamini_hochberg(pvals, alpha)
    return FamilyResult(m, alpha, bonf_t, pvals, bonf, fdr)
