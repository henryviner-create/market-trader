"""Cross-validation splitters that respect time.

Standard k-fold leaks: with overlapping label horizons, training samples adjacent
to the test fold share information with it. We provide:

* :func:`walk_forward` — rolling or expanding train/test windows in time order.
* :class:`PurgedKFold` — k-fold that **purges** training samples whose label
  horizon overlaps the test fold and **embargoes** a buffer immediately after it
  (López de Prado, *Advances in Financial Machine Learning*, ch. 7).

Inputs are assumed sorted in event order.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

IndexArray = NDArray[np.intp]


def walk_forward(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    *,
    expanding: bool = False,
) -> Iterator[tuple[IndexArray, IndexArray]]:
    """Yield ``(train_idx, test_idx)`` positional windows advancing through time."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if step is None:
        step = test_size
    if step <= 0:
        raise ValueError("step must be positive")

    indices = np.arange(n_samples, dtype=np.intp)
    pos = 0
    while True:
        train_end = train_size + pos
        test_end = train_end + test_size
        if test_end > n_samples:
            break
        train_start = 0 if expanding else pos
        yield indices[train_start:train_end], indices[train_end:test_end]
        pos += step


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 5
    embargo: float = 0.0  # fraction of n_samples embargoed after each test fold

    def split(self, t0: ArrayLike, t1: ArrayLike) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Split samples whose label spans ``[t0_i, t1_i]`` (event-ordered).

        ``t0`` is each sample's start time/position and ``t1`` the time/position at
        which its label is realised. Comparable scalars (ints or ``datetime64``)
        both work.
        """
        a0 = np.asarray(t0)
        a1 = np.asarray(t1)
        if a0.shape != a1.shape or a0.ndim != 1:
            raise ValueError("t0 and t1 must be 1-D arrays of equal length")
        if np.any(a1 < a0):
            raise ValueError("t1 must be >= t0 elementwise")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        n = a0.shape[0]
        indices = np.arange(n, dtype=np.intp)
        embargo_n = round(n * self.embargo)

        for test_idx in np.array_split(indices, self.n_splits):
            if test_idx.size == 0:
                continue
            test_start = a0[test_idx].min()
            test_end = a1[test_idx].max()

            # Purge: drop any training sample whose label horizon overlaps the test block.
            overlap = (a1 >= test_start) & (a0 <= test_end)
            train_mask = ~overlap
            train_mask[test_idx] = False

            # Embargo: drop a positional buffer immediately after the test block.
            if embargo_n > 0:
                right = int(test_idx.max()) + 1
                train_mask[right : right + embargo_n] = False

            yield indices[train_mask], test_idx
