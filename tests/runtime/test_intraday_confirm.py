"""The intraday confirmation gate — each anti-whipsaw rule, in isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_trader.runtime.intraday_confirm import ConfirmationConfig, IntradayConfirmation

T0 = datetime(2024, 6, 10, 15, 0, tzinfo=UTC)


def _gate(**kw) -> IntradayConfirmation:
    return IntradayConfirmation(ConfirmationConfig(**kw))


def _feed(gate, symbol, *, n, signal, vol=0.01, volume_ratio=2.0, daily=1.0, start=T0, step_min=5):
    """Feed ``n`` identical checks; return the last confirm() result."""
    out = 0
    for i in range(n):
        out = gate.confirm(
            symbol,
            signal=signal,
            recent_vol=vol,
            volume_ratio=volume_ratio,
            daily_signal=daily,
            now=start + timedelta(minutes=step_min * i),
        )
    return out


def test_single_spike_does_not_trade() -> None:
    # One significant bar is not enough — persistence guards against a spike that reverts.
    gate = _gate(persistence=3)
    assert _feed(gate, "AAA", n=1, signal=0.05) == 0


def test_persistent_significant_volume_confirmed_move_trades() -> None:
    gate = _gate(persistence=3)
    assert _feed(gate, "AAA", n=3, signal=0.05) == 1  # +ve, significant, persistent, confirmed


def test_sub_threshold_move_never_trades() -> None:
    # A move small relative to the name's own volatility is noise, however persistent.
    gate = _gate(persistence=3, sigma_mult=2.0)
    assert _feed(gate, "AAA", n=6, signal=0.01, vol=0.02) == 0  # 0.01 < 2*0.02


def test_thin_volume_blocks_the_trade() -> None:
    gate = _gate(persistence=3, volume_mult=1.5)
    assert _feed(gate, "AAA", n=4, signal=0.05, volume_ratio=1.0) == 0  # below volume confirmation


def test_contradicting_the_daily_signal_blocks_the_trade() -> None:
    gate = _gate(persistence=3)
    assert _feed(gate, "AAA", n=4, signal=0.05, daily=-1.0) == 0  # intraday +ve vs daily -ve


def test_direction_flip_resets_persistence() -> None:
    gate = _gate(persistence=3)
    gate.confirm("AAA", signal=0.05, recent_vol=0.01, volume_ratio=2.0, daily_signal=1.0, now=T0)
    gate.confirm(
        "AAA",
        signal=-0.05,  # flips
        recent_vol=0.01,
        volume_ratio=2.0,
        daily_signal=-1.0,
        now=T0 + timedelta(minutes=5),
    )
    out = gate.confirm(
        "AAA",
        signal=-0.05,
        recent_vol=0.01,
        volume_ratio=2.0,
        daily_signal=-1.0,
        now=T0 + timedelta(minutes=10),
    )
    assert out == 0  # only two in the new direction -> not yet persistent


def test_cooldown_blocks_a_second_action_until_it_elapses() -> None:
    gate = _gate(persistence=3, cooldown_minutes=30.0)
    assert _feed(gate, "AAA", n=3, signal=0.05) == 1  # first action at T0+10m
    # another confirmed signal 5 min later is inside the 30-min cooldown
    blocked = gate.confirm(
        "AAA",
        signal=0.05,
        recent_vol=0.01,
        volume_ratio=2.0,
        daily_signal=1.0,
        now=T0 + timedelta(minutes=15),
    )
    assert blocked == 0
    # well past the cooldown it acts again
    later = gate.confirm(
        "AAA",
        signal=0.05,
        recent_vol=0.01,
        volume_ratio=2.0,
        daily_signal=1.0,
        now=T0 + timedelta(minutes=60),
    )
    assert later == 1
