"""Broker-agnostic interface and order/position types.

The same interface backs the paper broker, the Alpaca adapter, and any future
broker — so strategy/execution code never changes between paper and live; only
the concrete broker, base URL, and keys differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class BrokerError(RuntimeError):
    """A broker rejected an order or a request failed. Broker-agnostic base so the
    execution engine can isolate a single bad order (e.g. a 403 ``insufficient buying
    power`` / ``not fractionable``) and skip it without aborting the whole rebalance.
    Concrete adapters (e.g. ``AlpacaError``) subclass this."""


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    NEW = "new"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    client_order_id: str  # client-generated => idempotent submission
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pl: float = 0.0


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    last_equity: float = 0.0  # equity at the previous close -> today's P&L = equity - last_equity


@runtime_checkable
class Broker(Protocol):
    def submit_order(self, order: Order) -> Order: ...

    def cancel_order(self, client_order_id: str) -> None: ...

    def get_positions(self) -> list[Position]: ...

    def get_open_orders(self) -> list[Order]: ...

    def get_account(self) -> Account: ...
