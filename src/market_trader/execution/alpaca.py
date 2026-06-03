"""Alpaca broker adapter (REST via stdlib urllib; no SDK dependency).

Defaults to the **paper** base URL. Paper and live share identical endpoints —
only ``base_url`` + keys differ — so going live is a config change, never a code
change. Live keys belong only in the secret manager, and live routing is gated by
``Settings.assert_live_allowed()`` upstream. Not exercised in tests (needs keys +
network); the in-memory :class:`PaperBroker` drives the test loop.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from market_trader.execution.broker import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
}


class AlpacaBroker:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        paper: bool = True,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca key_id and secret_key are required")
        self._base = base_url or (PAPER_BASE_URL if paper else LIVE_BASE_URL)
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "content-type": "application/json",
        }
        self._timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self._base + path, data=data, method=method, headers=self._headers
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else {}

    def submit_order(self, order: Order) -> Order:
        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": "day",
            "client_order_id": order.client_order_id,
        }
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        resp = self._request("POST", "/v2/orders", payload)
        order.status = _STATUS_MAP.get(resp.get("status", ""), OrderStatus.SUBMITTED)
        if resp.get("filled_qty"):
            order.filled_qty = float(resp["filled_qty"])
        if resp.get("filled_avg_price"):
            order.filled_avg_price = float(resp["filled_avg_price"])
        return order

    def cancel_order(self, client_order_id: str) -> None:
        found = self._request(
            "GET", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}"
        )
        if found.get("id"):
            self._request("DELETE", f"/v2/orders/{found['id']}")

    def get_positions(self) -> list[Position]:
        resp = self._request("GET", "/v2/positions")
        return [
            Position(
                p["symbol"],
                float(p["qty"]),
                float(p.get("avg_entry_price", 0.0)),
                market_value=float(p.get("market_value", 0.0)),
                unrealized_pl=float(p.get("unrealized_pl", 0.0)),
            )
            for p in resp
        ]

    def get_open_orders(self) -> list[Order]:
        resp = self._request("GET", "/v2/orders?status=open&limit=500&nested=false")
        return [
            Order(
                client_order_id=o.get("client_order_id", ""),
                symbol=o["symbol"],
                side=OrderSide(o["side"]),
                qty=float(o.get("qty") or 0.0),
                status=_STATUS_MAP.get(o.get("status", ""), OrderStatus.ACCEPTED),
                filled_qty=float(o.get("filled_qty") or 0.0),
            )
            for o in resp
        ]

    def get_clock(self) -> dict[str, Any]:
        """Alpaca market clock: {is_open, next_open, next_close, timestamp}."""
        return self._request("GET", "/v2/clock")

    def is_market_open(self) -> bool:
        return bool(self.get_clock().get("is_open", False))

    def get_account(self) -> Account:
        a = self._request("GET", "/v2/account")
        return Account(
            equity=float(a.get("equity", 0.0)),
            cash=float(a.get("cash", 0.0)),
            buying_power=float(a.get("buying_power", 0.0)),
            last_equity=float(a.get("last_equity", 0.0)),
        )
