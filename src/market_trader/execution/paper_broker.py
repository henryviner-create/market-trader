"""Deterministic in-memory paper broker.

Simulates immediate fills at a provided price (market) or marketable-limit fills,
tracks positions/cash, and is idempotent by ``client_order_id`` (re-submitting the
same id never double-fills). Drives the full execution loop in tests and offline
simulation with zero network and zero capital at risk.
"""

from __future__ import annotations

from market_trader.execution.broker import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class PaperBroker:
    def __init__(self, prices: dict[str, float], *, starting_cash: float = 100_000.0) -> None:
        self._prices = dict(prices)
        self._cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def submit_order(self, order: Order) -> Order:
        if order.client_order_id in self._orders:  # idempotent
            return self._orders[order.client_order_id]

        price = self._prices.get(order.symbol)
        if price is None or price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "no market price"
            self._orders[order.client_order_id] = order
            return order

        fill_price = price
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            marketable = (order.side == OrderSide.BUY and price <= order.limit_price) or (
                order.side == OrderSide.SELL and price >= order.limit_price
            )
            if not marketable:
                order.status = OrderStatus.ACCEPTED  # rests open
                self._orders[order.client_order_id] = order
                return order
            fill_price = order.limit_price

        signed = order.qty if order.side == OrderSide.BUY else -order.qty
        self._apply_fill(order.symbol, signed, fill_price)
        self._cash -= signed * fill_price
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = fill_price
        self._orders[order.client_order_id] = order
        return order

    def _apply_fill(self, symbol: str, signed_qty: float, price: float) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            self._positions[symbol] = Position(symbol, signed_qty, price)
            return
        new_qty = pos.qty + signed_qty
        if abs(new_qty) < 1e-12:
            del self._positions[symbol]
        elif (pos.qty > 0) == (signed_qty > 0):  # increasing the position
            avg = (pos.avg_price * pos.qty + price * signed_qty) / new_qty
            self._positions[symbol] = Position(symbol, new_qty, avg)
        else:  # reducing or flipping
            avg = price if (new_qty > 0) != (pos.qty > 0) else pos.avg_price
            self._positions[symbol] = Position(symbol, new_qty, avg)

    def cancel_order(self, client_order_id: str) -> None:
        order = self._orders.get(client_order_id)
        if order and order.status in (OrderStatus.NEW, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED):
            order.status = OrderStatus.CANCELLED

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> Account:
        market_value = sum(
            p.qty * self._prices.get(p.symbol, p.avg_price) for p in self._positions.values()
        )
        equity = self._cash + market_value
        return Account(equity=equity, cash=self._cash, buying_power=max(self._cash, 0.0))
