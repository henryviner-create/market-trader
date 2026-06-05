"""Intraday confirmation gate — don't trade on minor fluctuations.

A naive intraday loop re-forms the book on every minute-bar wiggle and dies by overtrading.
A real desk requires a candidate move to clear *several independent* confirmations before it
trades, each defending against a different source of false signal:

1. **Volatility-scaled significance** — the move must exceed ``sigma_mult x`` the name's own
   recent volatility, so a stock's normal jitter isn't mistaken for signal (a 1% move in a
   calm name is real; in a meme name it's noise).
2. **Persistence** — it must point the same way for ``persistence`` consecutive checks, so a
   one-bar spike that reverts is ignored.
3. **Volume confirmation** — the independent second data point: a move on elevated volume is
   real; on thin volume it's microstructure noise (``volume_ratio >= volume_mult``).
4. **Agreement with the slower (daily) signal** — don't let intraday noise fight the
   validated daily thesis.
5. **Per-name cooldown** — after acting, wait ``cooldown_minutes`` before acting on that name
   again, so it can't flip-flop.

The gate is stateful (it remembers each name's recent directions and last action across
loop passes) and returns ``+1`` / ``-1`` / ``0``; only a unanimous pass emits a non-zero
action. It is a *filter on conviction*, never an order itself — orders still go through the
validated scorer + risk pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ConfirmationConfig:
    persistence: int = 3  # consecutive same-direction checks required
    sigma_mult: float = 2.0  # |signal| must exceed this x the name's recent volatility
    volume_mult: float = 1.5  # recent volume must exceed this x typical
    require_daily_agreement: bool = True  # intraday action must not fight the daily signal
    cooldown_minutes: float = 30.0  # min gap between actions on one name


@dataclass
class _NameState:
    directions: deque[int]
    last_action: datetime | None = None


@dataclass
class IntradayConfirmation:
    """Stateful, per-name confirmation gate (see module docstring)."""

    config: ConfirmationConfig = field(default_factory=ConfirmationConfig)

    def __post_init__(self) -> None:
        self._state: dict[str, _NameState] = {}

    def confirm(
        self,
        symbol: str,
        *,
        signal: float,
        recent_vol: float,
        volume_ratio: float,
        daily_signal: float,
        now: datetime,
    ) -> int:
        """Return +1 (buy) / -1 (sell) / 0 (no action) after all confirmations.

        ``signal`` is the intraday conviction, ``recent_vol`` its typical magnitude (e.g. the
        name's recent return std), ``volume_ratio`` recent vs typical volume, ``daily_signal``
        the slower view's sign. Every check must pass; otherwise 0.
        """
        cfg = self.config
        st = self._state.setdefault(symbol, _NameState(deque(maxlen=max(1, cfg.persistence))))

        # 1. volatility-scaled significance (no vol estimate -> can't assess -> not significant)
        threshold = cfg.sigma_mult * max(float(recent_vol), 0.0)
        direction = (1 if signal > 0 else -1) if (threshold > 0 and abs(signal) >= threshold) else 0
        st.directions.append(direction)

        # 2. persistence: the last `persistence` checks all the same, non-zero direction
        if len(st.directions) < cfg.persistence:
            return 0
        proposed = st.directions[-1]
        if proposed == 0 or any(d != proposed for d in st.directions):
            return 0

        # 3. volume confirmation (the independent data point)
        if volume_ratio < cfg.volume_mult:
            return 0

        # 4. don't fight a present daily signal
        if cfg.require_daily_agreement and daily_signal != 0 and proposed * daily_signal < 0:
            return 0

        # 5. per-name cooldown
        if (
            st.last_action is not None
            and (now - st.last_action).total_seconds() < cfg.cooldown_minutes * 60.0
        ):
            return 0

        st.last_action = now
        return proposed
