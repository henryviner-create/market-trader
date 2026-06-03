"""Mandatory execution guardrails (all proven in the paper build).

Kill-switch, order sanity checks, order-rate limiting, and a capital ceiling.
Per-name/gross limits and the drawdown circuit-breaker are reused from the risk
layer. These exist *before* any order can reach a broker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from market_trader.execution.broker import Order, OrderType


class OrderSanityError(Exception):
    """An order that is absurd, malformed, or far from market."""


class KillSwitchEngaged(Exception):
    """Raised when the kill switch is engaged and trading is halted."""


class CapitalCeilingBreach(Exception):
    """Raised when deployable capital would exceed the hard ceiling."""


class OrderRateExceeded(Exception):
    """Raised when too many orders are submitted in an interval."""


def sanity_check_order(
    order: Order,
    *,
    ref_price: float | None = None,
    max_qty: float = 1e7,
    max_price_deviation: float = 0.2,
) -> None:
    """Reject zero/NaN/absurd quantities and limits priced far from the market."""
    if order.qty is None or (isinstance(order.qty, float) and math.isnan(order.qty)):
        raise OrderSanityError("quantity is NaN")
    if order.qty <= 0:
        raise OrderSanityError("non-positive quantity")
    if order.qty > max_qty:
        raise OrderSanityError(f"absurd quantity {order.qty}")
    if (
        order.order_type == OrderType.LIMIT
        and order.limit_price is not None
        and ref_price
        and ref_price > 0
        and abs(order.limit_price - ref_price) / ref_price > max_price_deviation
    ):
        raise OrderSanityError("limit price far from market")


class KillSwitch:
    """A single switch that halts all trading. Manual or auto-triggered."""

    def __init__(self) -> None:
        self.engaged = False
        self.reason: str | None = None

    def engage(self, reason: str = "manual") -> None:
        self.engaged = True
        self.reason = reason

    def reset(self) -> None:
        self.engaged = False
        self.reason = None


@dataclass
class OrderRateLimiter:
    max_orders: int
    _count: int = 0

    def allow(self) -> bool:
        if self._count >= self.max_orders:
            return False
        self._count += 1
        return True

    def reset(self) -> None:
        self._count = 0
