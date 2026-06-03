"""Signal pruning: retire signals whose rolling IC has decayed to noise."""

from __future__ import annotations

from collections.abc import Mapping


def prune_signals_by_ic(
    ic_by_signal: Mapping[str, float], *, min_abs_ic: float = 0.02
) -> tuple[list[str], list[str]]:
    """Return ``(kept, pruned)`` signal names by absolute rolling IC threshold."""
    kept = sorted(s for s, ic in ic_by_signal.items() if abs(ic) >= min_abs_ic)
    pruned = sorted(s for s, ic in ic_by_signal.items() if abs(ic) < min_abs_ic)
    return kept, pruned
