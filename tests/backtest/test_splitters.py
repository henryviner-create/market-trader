"""Unit tests for time-aware cross-validation splitters."""

from __future__ import annotations

import numpy as np

from market_trader.backtest.splitters import PurgedKFold, walk_forward


def test_walk_forward_rolling_windows() -> None:
    splits = list(walk_forward(10, train_size=4, test_size=2, step=2))
    assert len(splits) == 3
    assert splits[0][0].tolist() == [0, 1, 2, 3]
    assert splits[0][1].tolist() == [4, 5]
    assert splits[1][0].tolist() == [2, 3, 4, 5]  # rolling: fixed-size window slides
    assert splits[1][1].tolist() == [6, 7]


def test_walk_forward_expanding_train_grows() -> None:
    splits = list(walk_forward(10, train_size=4, test_size=2, step=2, expanding=True))
    assert splits[0][0].tolist() == [0, 1, 2, 3]
    assert splits[1][0].tolist() == [0, 1, 2, 3, 4, 5]  # expanding: train accumulates


def test_purged_kfold_purges_overlap_and_embargoes() -> None:
    n = 20
    t0 = np.arange(n)
    t1 = t0 + 2  # each label horizon spans 2 steps -> neighbours overlap
    pk = PurgedKFold(n_splits=4, embargo=0.1)  # embargo_n = round(20*0.1) = 2

    seen_test: list[int] = []
    for train_idx, test_idx in pk.split(t0, t1):
        seen_test.extend(test_idx.tolist())
        test_start = t0[test_idx].min()
        test_end = t1[test_idx].max()

        # No training sample's label horizon overlaps the test block.
        assert not np.any((t1[train_idx] >= test_start) & (t0[train_idx] <= test_end))
        # Train and test are disjoint.
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        # Embargoed positions immediately after the test block are excluded.
        right = int(test_idx.max()) + 1
        for pos in range(right, min(right + 2, n)):
            assert pos not in set(train_idx.tolist())

    # Folds partition all samples exactly once as test.
    assert sorted(seen_test) == list(range(n))
